"""Split each real-world teleop HDF5 demo into its sub-skill segments and convert each segment into
its own LeRobot-format episode, labeled with that sub-skill's own short prompt.

Input: a directory of HDF5 files written by the real teleop pipeline (env family
"Real-*-Franka-IK-Rel-v0", the real counterpart of SoftMimicGen's franka_towel sim env -- same obs
vocabulary, see convert_towel_hdf5_to_lerobot.py's module docstring), already clipped/exported by
zeyu/PHD_LAB/Amazon_Project/hdf5-lerobot-visualizer's web viewer (`demo_X.hdf5` -> `demo_X_clipped.hdf5`,
e.g. coffee_box_20261111_220249_clipped.hdf5), e.g.
/data/zeyu/PHD_LAB/Amazon_Project/real_world_demo/processed/*_clipped.hdf5:

    data/
      demo_N/                               (attrs: num_samples, source_demo, source_run, success)
        actions                  (T, 7)  float32  -- delta EEF pose (IK-relative, 6) + gripper (+1 open / -1 close)
        obs/
          agentview_image         (T, 512, 512, 3) uint8
          robot0_eye_in_hand_image (T, 512, 512, 3) uint8
          robot0_eef_pos           (T, 3)  float32
          robot0_eef_quat          (T, 4)  float32  -- (w, x, y, z)
          robot0_gripper_qpos      (T, 2)  float32  -- second finger stored negative

Requires `subskill_boundaries.json` alongside the HDF5 files -- written by the web viewer's manual
boundary editor (hdf5-lerobot-visualizer's /api/boundaries; see its subskill_boundaries.py), one entry
per HDF5 filename, then per demo key: a list of segments, each

    {"start_frame": int, "end_frame": int, "prompt": str, "skill_idx": int,
     "interaction_start_frame": int, "done_start_frame": int}

(both frame fields are absolute indices into that demo's own arrays, inclusive). Unlike
annotate_subskill_boundaries.py's auto-detected sim boundaries (which carry a per-frame phase_labels
array from --label-phases), these were hand-labeled frame-by-frame in the viewer, so there's only a
single split frame each for "moving" -> "interaction" phase and for "done": every frame from
interaction_start_frame onward (within the segment) gets phase=1, and every frame from
done_start_frame onward (plus all padding frames) gets subskill_done=1. A segment missing either field
is an error naming the offending file/demo/skill, not a silent skip -- same policy as the sim script.

Each segment gets `--pad-steps` (default 5) no-op frames appended after its last real frame: the
observation (images, proprioceptive state) is held at its last real value, the action is the all-zero
vector (commanded literally, not a per-dim "stay put" rule -- real OSC delta-pose control already
treats a zero action as "don't move", and the gripper's zero would re-command neither open nor close;
if that's ever wrong for a particular controller, re-check against this script's --action-type), and
subskill_done is always 1 regardless of the real segment's own done_start_frame.

Usage:
    uv run examples/franka_towel/convert_real_subskills_to_lerobot.py \\
        --data-dir /data/zeyu/PHD_LAB/Amazon_Project/real_world_demo/processed \\
        --repo-name real_world_subskills \\
        --output-dir /data/zeyu/PHD_LAB/Amazon_Project/lerobot_datasets/real_world_subskills

Then pass the same directory to `scripts/compute_norm_stats.py --local-root` and
`scripts/train.py --data.local-root`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro

ROOM_CAMERA = "agentview_image"
WRIST_CAMERA = "robot0_eye_in_hand_image"


def canonicalize_quat(quat: np.ndarray) -> np.ndarray:
    """Pick the sign of each (w, x, y, z) quaternion so that x >= 0 (q and -q are the same rotation) --
    same rule as convert_towel_hdf5_to_lerobot.py, needed because this gripper orientation also sits
    near a w~0 (180-degree-about-x) pose where the raw recording flips sign across demos/frames."""
    sign = np.where(quat[..., 1:2] < 0, -1.0, 1.0)
    return (quat * sign).astype(np.float32)


def _state(demo: h5py.Group) -> np.ndarray:
    obs = demo["obs"]
    return np.concatenate(
        [obs["robot0_eef_pos"][()], canonicalize_quat(obs["robot0_eef_quat"][()]), obs["robot0_gripper_qpos"][()]],
        axis=-1,
    ).astype(np.float32)


def _demo_sort_key(name: str) -> int:
    return int(name.split("_")[1])


def _read_fps(path: Path) -> float:
    with h5py.File(path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    sim = env_args.get("sim_args", {})
    return 1.0 / (float(sim["dt"]) * float(sim.get("decimation", 1)))


def main(
    data_dir: str,
    *,
    repo_name: str = "real_world_subskills",
    output_dir: str | None = None,
    boundaries_file: str | None = None,
    pad_steps: int = 5,
    image_size: int = 256,
    task_glob: str = "*_clipped.hdf5",
    only_success: bool = True,
    max_episodes_per_task: int | None = None,
    image_writer_processes: int = 5,
    image_writer_threads: int = 10,
    push_to_hub: bool = False,
):
    """Convert a directory of clipped real-world teleop HDF5 files into a LeRobot dataset of sub-skill
    episodes (see module docstring).

    Args:
        data_dir: Directory containing the clipped HDF5 files (e.g.
            real_world_demo/processed), one per recording session/task.
        repo_name: Dataset identifier, also used as the Hugging Face Hub repo id if --push-to-hub is
            set. If --output-dir isn't given, this ALSO determines where the dataset is written on
            disk ($HF_LEROBOT_HOME/<repo_name>).
        output_dir: If set, writes the dataset directly to this local directory instead of
            $HF_LEROBOT_HOME/<repo_name> -- pass this SAME directory to
            scripts/compute_norm_stats.py --local-root and scripts/train.py --data.local-root.
        boundaries_file: Path to subskill_boundaries.json. Defaults to
            <data_dir>/subskill_boundaries.json.
        pad_steps: Number of all-zero-action no-op frames appended to the end of every sub-skill
            segment (see module docstring). 0 disables padding.
        image_size: Images are resized (with padding) to image_size x image_size.
        task_glob: Glob pattern (relative to data_dir) selecting which HDF5 files to convert.
        only_success: Skip demos whose `success` attr is false.
        max_episodes_per_task: If set, only convert the first N demos of each HDF5 file.
        image_writer_processes / image_writer_threads: LeRobot's PNG-encoding workers.
        push_to_hub: Whether to push the resulting dataset to the Hugging Face Hub.
    """
    dir_path = Path(data_dir)
    task_paths = sorted(dir_path.glob(task_glob))
    if not task_paths:
        raise FileNotFoundError(f"No files matching {task_glob!r} found in {dir_path}")
    b_path = Path(boundaries_file) if boundaries_file else dir_path / "subskill_boundaries.json"
    if not b_path.is_file():
        raise FileNotFoundError(f"{b_path} not found -- label sub-skill boundaries in the web viewer first.")
    all_boundaries = json.loads(b_path.read_text())

    output_path = Path(output_dir) if output_dir else HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    fps_values = {round(_read_fps(p)) for p in task_paths}
    if len(fps_values) > 1:
        raise ValueError(f"task files disagree on control frequency: {fps_values}")
    fps = fps_values.pop()

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_dir,
        robot_type="panda",
        fps=fps,
        features={
            "image": {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            # eef_pos (3) + canonicalized eef_quat w,x,y,z (4) + gripper_qpos (2)
            "state": {"dtype": "float32", "shape": (9,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            # 1 from done_start_frame onward (inclusive) plus every padding frame, 0 before.
            "subskill_done": {"dtype": "int64", "shape": (1,), "names": ["subskill_done"]},
            # 0 ("moving") before interaction_start_frame, 1 ("interaction") from it onward.
            "phase": {"dtype": "int64", "shape": (1,), "names": ["phase"]},
            # "actions" ++ "phase" ++ "subskill_done" concatenated -- see
            # convert_libero_subskills_to_lerobot.py's module docstring for why this rides alongside
            # the three features above instead of replacing them.
            "actions_with_aux": {"dtype": "float32", "shape": (9,), "names": ["actions_with_aux"]},
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    num_episodes_written = 0
    num_demos_skipped = 0
    episode_source: list[dict] = []  # provenance of every written episode -> meta/episode_source.jsonl

    for task_path in tqdm.tqdm(task_paths, desc="task files"):
        task_boundaries = all_boundaries.get(task_path.name)
        if task_boundaries is None:
            print(f"warning: no boundaries for {task_path.name} in {b_path}, skipping this task file")
            continue

        with h5py.File(task_path, "r") as f:
            data = f["data"]
            demo_keys = sorted(data.keys(), key=_demo_sort_key)
            if only_success:
                demo_keys = [k for k in demo_keys if bool(data[k].attrs.get("success", True))]
            demo_keys = demo_keys[:max_episodes_per_task]

            for demo_key in tqdm.tqdm(demo_keys, desc=task_path.stem, leave=False):
                demo_entry = task_boundaries["demos"].get(demo_key)
                if demo_entry is None:
                    num_demos_skipped += 1
                    continue

                demo = data[demo_key]
                actions = demo["actions"][()].astype(np.float32)
                states = _state(demo)
                room = image_tools.resize_with_pad(demo["obs"][ROOM_CAMERA][()], image_size, image_size)
                wrist = image_tools.resize_with_pad(demo["obs"][WRIST_CAMERA][()], image_size, image_size)

                for boundary in demo_entry["boundaries"]:
                    start, end, prompt = boundary["start_frame"], boundary["end_frame"], boundary["prompt"]
                    skill_idx = boundary["skill_idx"]
                    interaction_start = boundary.get("interaction_start_frame")
                    done_start = boundary.get("done_start_frame")
                    if interaction_start is None or done_start is None:
                        raise ValueError(
                            f"{task_path.name}/{demo_key} skill {skill_idx} ({prompt!r}) is missing "
                            f"interaction_start_frame/done_start_frame -- label the moving/interaction "
                            f"and task-done splits in the web viewer before converting."
                        )

                    not_done = np.array([0], dtype=np.int64)
                    done = np.array([1], dtype=np.int64)
                    last_phase = None
                    for i in range(start, end + 1):
                        phase = np.array([1 if i >= interaction_start else 0], dtype=np.int64)
                        subskill_done = done if i >= done_start else not_done
                        last_phase = phase
                        dataset.add_frame(
                            {
                                "image": room[i],
                                "wrist_image": wrist[i],
                                "state": states[i],
                                "actions": actions[i],
                                "task": prompt,
                                "subskill_done": subskill_done,
                                "phase": phase,
                                "actions_with_aux": np.concatenate(
                                    [actions[i], phase.astype(np.float32), subskill_done.astype(np.float32)]
                                ),
                            }
                        )
                    if pad_steps > 0:
                        pad_image, pad_wrist, pad_state = room[end], wrist[end], states[end]
                        pad_action = np.zeros_like(actions[end])
                        pad_action_with_aux = np.concatenate(
                            [pad_action, last_phase.astype(np.float32), done.astype(np.float32)]
                        )
                        for _ in range(pad_steps):
                            dataset.add_frame(
                                {
                                    "image": pad_image,
                                    "wrist_image": pad_wrist,
                                    "state": pad_state,
                                    "actions": pad_action,
                                    "task": prompt,
                                    "subskill_done": done,
                                    "phase": last_phase,
                                    "actions_with_aux": pad_action_with_aux,
                                }
                            )
                    dataset.save_episode()
                    episode_source.append(
                        {
                            "episode_index": num_episodes_written,
                            "hdf5": task_path.name,
                            "demo": demo_key,
                            "skill_idx": skill_idx,
                            "prompt": prompt,
                            "start_frame": start,
                            "end_frame": end,
                            "padded_frames": pad_steps,
                            "interaction_start_frame": interaction_start,
                            "done_start_frame": done_start,
                        }
                    )
                    num_episodes_written += 1

    (output_path / "meta").mkdir(parents=True, exist_ok=True)
    (output_path / "meta" / "episode_source.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episode_source))
    prompts = sorted({r["prompt"] for r in episode_source})
    print(
        f"Wrote {num_episodes_written} sub-skill episodes across {len(prompts)} sub-skill(s) "
        f"({num_demos_skipped} demos skipped, not in {b_path.name})"
    )
    for p in prompts:
        print(f"  - {p}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["real-world", "panda", "hdf5", "subskills"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
