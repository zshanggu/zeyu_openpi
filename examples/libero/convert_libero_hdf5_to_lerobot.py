"""Convert raw LIBERO HDF5 demonstration files to the LeRobot dataset format.

The upstream `convert_libero_data_to_lerobot.py` script in this directory converts the
*RLDS/TFDS* re-release of LIBERO (`openvla/modified_libero_rlds`). This script instead
converts the *original* LIBERO benchmark HDF5 files (the `libero_10`, `libero_90`,
`libero_spatial`, `libero_object`, `libero_goal` directories you get from
https://github.com/Lifelong-Robot-Learning/LIBERO or the raw `libero_100.zip` download),
which look like:

    some_task_demo.hdf5
      data/                         (attrs: env_args, problem_info.language_instruction, ...)
        demo_0/
          actions          (T, 7)   float64   -- already delta actions (OSC_POSE, control_delta=True)
          obs/
            agentview_rgb        (T, 128, 128, 3) uint8
            eye_in_hand_rgb      (T, 128, 128, 3) uint8
            ee_states            (T, 6)  float64
            gripper_states       (T, 2)  float64
        demo_1/
          ...

Each *.hdf5 file is one task (one language instruction) containing many demonstrations.

Usage:
    uv run examples/libero/convert_libero_hdf5_to_lerobot.py \
        --data-dir /path/to/libero_90 \
        --repo-name your_hf_username/libero_90

If you want to push your dataset to the Hugging Face Hub, add --push-to-hub.

Note: to run the script, you need to install h5py: `uv pip install h5py`.
The resulting dataset is saved to the $HF_LEROBOT_HOME directory.
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
    language_instruction: str
    control_freq: float
    demo_keys: list[str]


def _read_task_file(path: Path) -> TaskFile:
    with h5py.File(path, "r") as f:
        attrs = f["data"].attrs
        env_args = json.loads(attrs["env_args"])
        problem_info = json.loads(attrs["problem_info"])
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        return TaskFile(
            path=path,
            language_instruction=problem_info["language_instruction"],
            control_freq=float(env_args["env_kwargs"].get("control_freq", 20)),
            demo_keys=demo_keys,
        )


def _flip(img: np.ndarray) -> np.ndarray:
    # LIBERO renders camera images with MuJoCo's OpenGL offscreen renderer, whose row
    # order is bottom-to-top, so frames need a vertical flip to display/train right-side-up.
    return np.ascontiguousarray(img[:, ::-1])


def main(
    data_dir: str,
    *,
    repo_name: str = "your_hf_username/libero_90",
    image_size: int = 128,
    max_episodes_per_task: int | None = None,
    task_glob: str = "*.hdf5",
    push_to_hub: bool = False,
):
    """Convert a directory of raw LIBERO *_demo.hdf5 files to a LeRobot dataset.

    Args:
        data_dir: Directory containing LIBERO *_demo.hdf5 files (e.g. libero_90, libero_10).
        repo_name: Output dataset name, also used as the Hugging Face Hub repo id if
            --push-to-hub is set.
        image_size: Camera images are stored at their native resolution (128x128 for the
            standard LIBERO release); openpi's model transforms resize to the model's input
            size at train/inference time regardless, so this normally doesn't need changing.
        max_episodes_per_task: If set, only convert the first N demos of each task file.
            Useful for a quick smoke test before converting the full dataset.
        task_glob: Glob pattern (relative to data_dir) selecting which *.hdf5 files to convert.
        push_to_hub: Whether to push the resulting dataset to the Hugging Face Hub.
    """
    data_dir_path = Path(data_dir)
    task_paths = sorted(data_dir_path.glob(task_glob))
    if not task_paths:
        raise FileNotFoundError(f"No files matching {task_glob!r} found in {data_dir_path}")

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=20,
        features={
            "image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            # 8-dim proprioceptive state: end-effector pose (position + orientation, 6)
            # plus gripper finger positions (2). Matches the state layout expected by
            # `openpi.policies.libero_policy.LiberoInputs` / the "physical-intelligence/libero"
            # reference dataset. openpi pads/truncates state to the model's dimension, so
            # exact size isn't load-bearing, but matching keeps norm stats comparable.
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for task_path in tqdm.tqdm(task_paths, desc="tasks"):
        task = _read_task_file(task_path)
        demo_keys = task.demo_keys[:max_episodes_per_task]

        with h5py.File(task.path, "r") as f:
            data = f["data"]
            for demo_key in tqdm.tqdm(demo_keys, desc=task_path.stem, leave=False):
                demo = data[demo_key]
                obs = demo["obs"]

                actions = demo["actions"][()].astype(np.float32)
                ee_states = obs["ee_states"][()]
                gripper_states = obs["gripper_states"][()]
                states = np.concatenate([ee_states, gripper_states], axis=-1).astype(np.float32)
                agentview = obs["agentview_rgb"][()]
                eye_in_hand = obs["eye_in_hand_rgb"][()]

                num_steps = actions.shape[0]
                for i in range(num_steps):
                    dataset.add_frame(
                        {
                            "image": _flip(agentview[i]),
                            "wrist_image": _flip(eye_in_hand[i]),
                            "state": states[i],
                            "actions": actions[i],
                            "task": task.language_instruction,
                        }
                    )
                dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
