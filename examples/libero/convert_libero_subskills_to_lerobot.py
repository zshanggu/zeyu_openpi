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
        env_args = json.loads(f["data"].attrs["env_args"])
        return TaskFile(path=path, control_freq=float(env_args["env_kwargs"].get("control_freq", 20)))


def _flip(img: np.ndarray) -> np.ndarray:
    # LIBERO renders camera images with MuJoCo's OpenGL offscreen renderer, whose row
    # order is bottom-to-top, so frames need a vertical flip to display/train right-side-up.
    return np.ascontiguousarray(img[:, ::-1])


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
    boundaries_file: str | None = None,
    pad_steps: int = 10,
    action_type: str = "delta",
    image_size: int = 128,
    max_episodes_per_task: int | None = None,
    task_glob: str = "*.hdf5",
    push_to_hub: bool = False,
):
    """Convert a directory of raw LIBERO *_demo.hdf5 files into a LeRobot
    dataset of sub-skill episodes (see module docstring).

    Args:
        data_dir: Directory containing LIBERO *_demo.hdf5 files (e.g. libero_10).
        repo_name: Output dataset name, also used as the Hugging Face Hub repo id if
            --push-to-hub is set.
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
        push_to_hub: Whether to push the resulting dataset to the Hugging Face Hub.
    """
    data_dir_path = Path(data_dir)
    task_paths = sorted(data_dir_path.glob(task_glob))
    if not task_paths:
        raise FileNotFoundError(f"No files matching {task_glob!r} found in {data_dir_path}")

    boundaries_path = Path(boundaries_file) if boundaries_file else data_dir_path / "subskill_boundaries.json"
    if not boundaries_path.is_file():
        raise FileNotFoundError(
            f"{boundaries_path} not found -- run annotate_subskill_boundaries.py on {data_dir_path} first."
        )
    all_boundaries = json.loads(boundaries_path.read_text())

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
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
        image_writer_threads=10,
        image_writer_processes=5,
    )

    num_episodes_written = 0
    num_demos_skipped = 0
    for task_path in tqdm.tqdm(task_paths, desc="tasks"):
        task = _read_task_file(task_path)
        task_boundaries = all_boundaries.get(task_path.name)
        if task_boundaries is None:
            print(f"warning: no boundaries for {task_path.name} in {boundaries_path}, skipping this task file")
            continue

        with h5py.File(task.path, "r") as f:
            data = f["data"]
            demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[:max_episodes_per_task]

            for demo_key in tqdm.tqdm(demo_keys, desc=task_path.stem, leave=False):
                demo_entry = task_boundaries["demos"].get(demo_key)
                if demo_entry is None:
                    num_demos_skipped += 1
                    continue

                demo = data[demo_key]
                obs = demo["obs"]
                actions = demo["actions"][()].astype(np.float32)
                states = np.concatenate([obs["ee_states"][()], obs["gripper_states"][()]], axis=-1).astype(np.float32)
                agentview = obs["agentview_rgb"][()]
                eye_in_hand = obs["eye_in_hand_rgb"][()]

                for boundary in demo_entry["boundaries"]:
                    start, end = boundary["start_frame"], boundary["end_frame"]
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
                    num_episodes_written += 1

    print(f"Wrote {num_episodes_written} sub-skill episodes ({num_demos_skipped} demos skipped, not in {boundaries_path.name})")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "hdf5", "subskills"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
