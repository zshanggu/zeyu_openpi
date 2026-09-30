"""Split each LIBERO HDF5 demo into its sub-skill segments and convert each
segment into its own LeRobot-format episode, labeled with that sub-skill's
own short prompt instead of the full long-horizon instruction.

Requires `subskill_boundaries.json` (see `annotate_subskill_boundaries.py`)
for the same HDF5 directory: that file records, per demo, the
[start_frame, end_frame] span and short language prompt for each sub-skill
(e.g. "pick up the alphabet soup", "place the alphabet soup in the basket").
This script does the actual cutting -- annotate_subskill_boundaries.py
deliberately stops at recording boundaries, doesn't touch the data itself.

Why pad each segment's end at all: openpi/LeRobot training samples each
timestep's target action *chunk* as the next `action_horizon` actions found
in the SAME episode -- a timestep needs that many real follow-up frames to
get a full, undistorted training target. A sub-skill segment ends at an
artificial mid-demo cut (not the natural end of the original episode, except
for the last segment), so its last `action_horizon - 1` frames would
otherwise come up short. This appends `--pad-steps` (default 10, matching
`pi05_libero`'s `action_horizon` in src/openpi/training/config.py) no-op
frames to the end of every segment: the observation (images, proprioceptive
state) is held at its last real value (nothing actually moves), and the
padding action means "stay exactly where you are" -- which differs by action
representation:

  - delta actions (this dataset's actual convention -- see
    convert_libero_hdf5_to_lerobot.py's docstring, confirmed directly
    against the raw HDF5 actions array): a zero-delta action for the 6 pose
    dims, since "no incremental motion" *is* "stay put" in a delta action
    space. The 7th (gripper) channel is NOT a delta here even though it
    rides along in the same array -- LIBERO's gripper channel is a held
    open(-1)/close(+1) command (confirmed empirically: real action arrays
    show it pinned at -1 or +1 for many consecutive steps, never smoothly
    varying the way the 6 pose deltas do), so zeroing it would spuriously
    command "half open" during the pad. It's held at its last real value
    instead.
  - absolute actions: the entire last real action repeated verbatim, since
    re-commanding the same absolute target produces zero further motion.

`--action-type` selects which rule applies (defaults to "delta", matching
this dataset -- pass "absolute" only if converting some other LIBERO-like
HDF5 source that isn't delta-action).

Usage:
    uv run examples/libero/convert_libero_subskills_to_lerobot.py \
        --data-dir /path/to/libero_10 \
        --repo-name your_hf_username/libero_10_subskills

    # OpenVLA-regenerated 256px data with randomized-distractor episodes (libero_modified_hdf5/...):
    # standard demos and randomized ones into two separate datasets
    uv run examples/libero/convert_libero_subskills_to_lerobot.py \
        --data-dir /path/to/libero90_openvla_256 --image-size 256 --skip-detection-failed \
        --repo-name libero90_lilo22_256 --output-dir /path/out/libero90_openvla_256_lilo22
    ... --groups distractor --output-dir /path/out/libero90_openvla_256_lilo22_distractor

    # Several source folders merged into ONE dataset (standard episodes first, then the randomized ones):
    uv run examples/libero/convert_libero_subskills_to_lerobot.py \
        --data-dir /path/to/libero90_openvla_256 \
        --extra-data-dirs /path/to/libero90_openvla_256_wooden_cabinet \
        --groups both --image-size 256 --skip-detection-failed \
        --repo-name libero90_lilo22_256 --output-dir /path/out/libero90_lilo22_256
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
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class TaskFile:
    path: Path
    control_freq: float


def _read_task_file(path: Path) -> TaskFile:
    with h5py.File(path, "r") as f:
        # Raw LIBERO files carry env_args; OpenVLA-regenerated ones (libero_modified_hdf5/*) carry no
        # attributes beyond problem_info. Nothing downstream needs the frequency (fps is fixed at 20).
        attrs = f["data"].attrs
        control_freq = 20.0
        if "env_args" in attrs:
            control_freq = float(json.loads(attrs["env_args"])["env_kwargs"].get("control_freq", 20))
        return TaskFile(path=path, control_freq=control_freq)


def _demo_sort_key(name: str) -> tuple:
    """demo_<N> -> (N,); distractor_<N>_<K> -> (N, K)."""
    return tuple(int(x) for x in name.split("_")[1:] if x.isdigit())


def _flip(img: np.ndarray) -> np.ndarray:
    # Raw HDF5 agentview_rgb/eye_in_hand_rgb arrays are the direct, unmodified
    # obs["agentview_image"]/obs["robot0_eye_in_hand_image"] MuJoCo render buffers --
    # third_party/libero/scripts/create_dataset.py stores them with no flip of its own
    # (agentview_images.append(obs["agentview_image"])), so they're in the exact same raw
    # orientation examples/libero/main.py corrects at inference time via
    # obs["agentview_image"][::-1, ::-1] (both axes, not just one -- a single-axis flip
    # left training images mirrored relative to what the model actually sees at eval time).
    return np.ascontiguousarray(img[::-1, ::-1])


def _pad_action(last_action: np.ndarray, action_type: str) -> np.ndarray:
    if action_type == "absolute":
        return last_action.copy()
    if action_type == "delta":
        pad = np.zeros_like(last_action)
        pad[6] = last_action[6]  # hold the gripper's last commanded open/close state, not a delta
        return pad
    raise ValueError(f"Unknown action_type {action_type!r} (expected 'delta' or 'absolute')")


def main(
    data_dir: str,
    *,
    repo_name: str = "your_hf_username/libero_10_subskills",
    output_dir: str | None = None,
    boundaries_file: str | None = None,
    pad_steps: int = 10,
    action_type: str = "delta",
    image_size: int = 128,
    max_episodes_per_task: int | None = None,
    task_glob: str = "*.hdf5",
    groups: str = "standard",
    skip_detection_failed: bool = False,
    extra_data_dirs: tuple[str, ...] = (),
    image_writer_processes: int = 5,
    image_writer_threads: int = 10,
    push_to_hub: bool = False,
):
    """Convert a directory of raw LIBERO *_demo.hdf5 files into a LeRobot
    dataset of sub-skill episodes (see module docstring).

    Args:
        data_dir: Directory containing LIBERO *_demo.hdf5 files (e.g. libero_10).
        repo_name: Dataset identifier, also used as the Hugging Face Hub repo id if
            --push-to-hub is set. If --output-dir isn't given, this ALSO determines where the
            dataset is written on disk ($HF_LEROBOT_HOME/<repo_name>).
        output_dir: If set, writes the dataset directly to this local directory instead of
            $HF_LEROBOT_HOME/<repo_name> -- pass this SAME directory to
            `scripts/compute_norm_stats.py --local-root` and `scripts/train.py
            --data.local-root` so all three steps agree on one single path, without needing to
            juggle repo_id/$HF_LEROBOT_HOME at all.
        boundaries_file: Path to subskill_boundaries.json. Defaults to
            `<data_dir>/subskill_boundaries.json` (where
            annotate_subskill_boundaries.py writes it by default).
        pad_steps: Number of no-op frames appended to the end of every sub-skill segment
            (see module docstring for why). 0 disables padding.
        action_type: "delta" (default, matches this dataset -- see module docstring) or
            "absolute". Determines what a "stay put" padding action looks like.
        image_size: Camera images are stored at their native resolution (128x128 for the
            standard LIBERO release); openpi's model transforms resize to the model's input
            size at train/inference time regardless, so this normally doesn't need changing.
        max_episodes_per_task: If set, only convert the first N demos of each task file.
        task_glob: Glob pattern (relative to data_dir) selecting which *.hdf5 files to convert.
        groups: Which episodes to convert. "standard" (default, the original behaviour): the `data` group.
            "distractor": the `data_distractor` group written by libero_rerender_script/rerender_libero90.py
            (the same demos replayed with the non-task objects randomized). "both": the two together.
            Convert them into SEPARATE datasets (run twice) if the randomized episodes must not be mixed
            with the standard ones -- LeRobot episodes carry no group label; every episode's origin is
            recorded in `meta/episode_source.jsonl` regardless.
        extra_data_dirs: More HDF5 directories to merge into the SAME output dataset (each with its own
            subskill_boundaries.json). Episodes are written group by group, then source by source
            (data_dir first, then these in order): with --groups both, every standard episode of every
            source comes first, then every randomized one, so each group is one contiguous episode range;
            `meta/merge_info.json` records the ranges and `meta/episode_source.jsonl` every episode's origin.
        image_writer_processes / image_writer_threads: LeRobot's PNG-encoding workers (encoding is the
            bottleneck; raise the process count on a many-core machine).
        skip_detection_failed: Drop a segment whose boundary was a fallback guess
            (`detection_failed` in subskill_boundaries.json) instead of converting it with a warning.
        push_to_hub: Whether to push the resulting dataset to the Hugging Face Hub.
    """
    if extra_data_dirs and boundaries_file:
        raise ValueError("--boundaries-file can only be used with a single data directory")

    @dataclasses.dataclass
    class Source:
        dir: Path
        task_paths: list
        boundaries: dict
        boundaries_path: Path

    sources: list[Source] = []
    for d in (data_dir, *extra_data_dirs):
        dir_path = Path(d)
        paths = sorted(dir_path.glob(task_glob))
        if not paths:
            raise FileNotFoundError(f"No files matching {task_glob!r} found in {dir_path}")
        b_path = Path(boundaries_file) if boundaries_file else dir_path / "subskill_boundaries.json"
        if not b_path.is_file():
            raise FileNotFoundError(f"{b_path} not found -- run annotate_subskill_boundaries.py on {dir_path} first.")
        sources.append(Source(dir_path, paths, json.loads(b_path.read_text()), b_path))
    boundaries_path = sources[0].boundaries_path  # only used in the summary message below

    output_path = Path(output_dir) if output_dir else HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_dir,
        robot_type="panda",
        fps=20,
        features={
            "image": {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            # 8-dim proprioceptive state: end-effector pose (position + orientation, 6)
            # plus gripper finger positions (2). Matches the state layout expected by
            # `openpi.policies.libero_policy.LiberoInputs` / the "physical-intelligence/libero"
            # reference dataset.
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    num_episodes_written = 0
    num_demos_skipped = 0
    num_segments_skipped_failed = 0
    episode_source: list[dict] = []  # provenance of every written episode -> meta/episode_source.jsonl
    wanted_groups = {"standard": ["data"], "distractor": ["data_distractor"], "both": ["data", "data_distractor"]}[groups]
    work = [(g, src, tp) for g in wanted_groups for src in sources for tp in src.task_paths]  # group-major order
    for group_name, source, task_path in tqdm.tqdm(work, desc="task files"):
        task = _read_task_file(task_path)
        task_boundaries = source.boundaries.get(task_path.name)
        if task_boundaries is None:
            print(f"warning: no boundaries for {task_path.name} in {source.boundaries_path}, skipping this task file")
            continue

        with h5py.File(task.path, "r") as f:
            demo_refs = []  # (group name, demo key)
            if group_name in f:
                keys = sorted(f[group_name].keys(), key=_demo_sort_key)[:max_episodes_per_task]
                demo_refs += [(group_name, k) for k in keys]

            for group_name, demo_key in tqdm.tqdm(demo_refs, desc=task_path.stem, leave=False):
                demo_entry = task_boundaries["demos"].get(demo_key)
                if demo_entry is None:
                    num_demos_skipped += 1
                    continue

                demo = f[group_name][demo_key]
                obs = demo["obs"]
                actions = demo["actions"][()].astype(np.float32)
                states = np.concatenate([obs["ee_states"][()], obs["gripper_states"][()]], axis=-1).astype(np.float32)
                agentview = obs["agentview_rgb"][()]
                eye_in_hand = obs["eye_in_hand_rgb"][()]

                for boundary in demo_entry["boundaries"]:
                    start, end = boundary["start_frame"], boundary["end_frame"]
                    if boundary.get("detection_failed") and skip_detection_failed:
                        num_segments_skipped_failed += 1
                        continue
                    if boundary.get("detection_failed"):
                        print(
                            f"warning: {task_path.name}/{demo_key} skill {boundary['skill_idx']} "
                            f"({boundary['prompt']!r}) boundary is a fallback guess, not a confirmed detection"
                        )

                    for i in range(start, end + 1):
                        dataset.add_frame(
                            {
                                "image": _flip(agentview[i]),
                                "wrist_image": _flip(eye_in_hand[i]),
                                "state": states[i],
                                "actions": actions[i],
                                "task": boundary["prompt"],
                            }
                        )
                    if pad_steps > 0:
                        pad_image, pad_wrist, pad_state = _flip(agentview[end]), _flip(eye_in_hand[end]), states[end]
                        pad_action = _pad_action(actions[end], action_type)
                        for _ in range(pad_steps):
                            dataset.add_frame(
                                {
                                    "image": pad_image,
                                    "wrist_image": pad_wrist,
                                    "state": pad_state,
                                    "actions": pad_action,
                                    "task": boundary["prompt"],
                                }
                            )
                    dataset.save_episode()
                    episode_source.append(
                        {
                            "episode_index": num_episodes_written,
                            "source_dir": str(source.dir),
                            "hdf5": task_path.name,
                            "group": group_name,
                            "demo": demo_key,
                            "skill_idx": boundary["skill_idx"],
                            "prompt": boundary["prompt"],
                            "start_frame": start,
                            "end_frame": end,
                            "padded_frames": pad_steps,
                        }
                    )
                    num_episodes_written += 1

    (output_path / "meta").mkdir(parents=True, exist_ok=True)
    merge_info: dict = {"sources": [str(src.dir) for src in sources], "groups": {}, "by_source_and_group": {}}
    for r in episode_source:
        g = merge_info["groups"].setdefault(r["group"], {"episodes": 0, "first_episode_index": r["episode_index"], "last_episode_index": r["episode_index"]})
        g["episodes"] += 1
        g["last_episode_index"] = r["episode_index"]
        key = f"{r['source_dir']}::{r['group']}"
        merge_info["by_source_and_group"][key] = merge_info["by_source_and_group"].get(key, 0) + 1
    (output_path / "meta" / "merge_info.json").write_text(json.dumps(merge_info, indent=2))
    (output_path / "meta" / "episode_source.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episode_source))
    print(
        f"Wrote {num_episodes_written} sub-skill episodes ({num_demos_skipped} demos skipped, not in {boundaries_path.name}"
        f"; {num_segments_skipped_failed} segments skipped for failed detection; {len(sources)} source dir(s))"
    )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "hdf5", "subskills"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
