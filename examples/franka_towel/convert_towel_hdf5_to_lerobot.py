"""Convert a SoftMimicGen Franka towel HDF5 file to the LeRobot dataset format.

Input: the HDF5 written by SoftMimicGen's experiments/policy_data pipeline (env
`Isaac-Fold-Towel-Franka-IK-Rel-Mimic-v0`, 20 Hz = sim dt 0.01 x decimation 5), e.g.
/data/hyeon/share/franka_towel_n50_seed1_20260926_1813/hdf5/franka_towel_n50_seed1.hdf5:

    data/                                   (attrs: env_args, total -- no language instruction)
      demo_N/                               (attrs: num_samples, success)
        actions                  (T, 7)  float32  -- delta EEF pose (IK-relative, 6) + gripper (+1 open / -1 close)
        obs/
          actions                (T, 7)  float32  -- the PREVIOUS action (zeros at t=0); not used
          agentview_image        (T, 512, 512, 3) uint8
          robot0_eye_in_hand_image (T, 512, 512, 3) uint8
          robot0_eef_pos         (T, 3)  float32
          robot0_eef_quat        (T, 4)  float32  -- (w, x, y, z)
          robot0_gripper_qpos    (T, 2)  float32  -- second finger is stored negative
          ...depth / normals / instance / geoedge (not converted)
        states/articulation/robot/joint_position (T, 9) float32  -- 7 arm joints + 2 fingers

Differences from examples/libero/convert_libero_hdf5_to_lerobot.py:
  * Images are NOT flipped. Isaac Lab camera renders are already upright (checked visually), unlike
    LIBERO's raw MuJoCo buffers. The eval client must send them unflipped too.
  * Images are resized (with padding, same as openpi_client.image_tools at inference) to --image-size.
  * The prompt is fixed (--prompt), since the HDF5 has no language instruction.
  * The EEF quaternion is sign-canonicalized. The gripper points straight down, so w ~ 0 and the stored
    quaternion flips between q and -q (6 times in the n50 seed1 file); we pick the sign with x >= 0,
    which is continuous for this task. Apply the same rule in the eval client.

State layouts (--state):
  "eef"   (default, 9): eef_pos (3) + canonical eef_quat w,x,y,z (4) + gripper_qpos (2)
  "joint" (9):          absolute joint positions, 7 arm joints + 2 fingers

Usage:
    uv run examples/franka_towel/convert_towel_hdf5_to_lerobot.py \
        --hdf5 /data/hyeon/share/franka_towel_n50_seed1_20260926_1813/hdf5/franka_towel_n50_seed1.hdf5 \
        --output-dir /data/zeyu/PHD_LAB/Deformable/lerobot_datasets/franka_towel_n50_seed1_256

Several HDF5 files (same env and camera layout) can be merged into one dataset: pass them all to
--hdf5, e.g. `--hdf5 a.hdf5 b.hdf5`. Their episodes are written file by file, in the given order.

Then pass the same directory to `scripts/compute_norm_stats.py --local-root` (or rely on the
`pi05_franka_towel_*` configs' local_root) and train.

Note: to run the script, you need h5py: `uv pip install h5py`.
"""

from __future__ import annotations

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
    """Pick the sign of each (w, x, y, z) quaternion so that x >= 0 (q and -q are the same rotation)."""
    sign = np.where(quat[..., 1:2] < 0, -1.0, 1.0)
    return (quat * sign).astype(np.float32)


def _state(demo: h5py.Group, layout: str) -> np.ndarray:
    if layout == "eef":
        obs = demo["obs"]
        return np.concatenate(
            [obs["robot0_eef_pos"][()], canonicalize_quat(obs["robot0_eef_quat"][()]), obs["robot0_gripper_qpos"][()]],
            axis=-1,
        ).astype(np.float32)
    if layout == "joint":
        return demo["states/articulation/robot/joint_position"][()].astype(np.float32)
    raise ValueError(f"Unknown state layout {layout!r} (use 'eef' or 'joint')")


def main(
    hdf5: list[str],
    *,
    repo_name: str = "franka_towel",
    output_dir: str | None = None,
    prompt: str = "fold the towel",
    state: str = "eef",
    image_size: int = 256,
    max_episodes: int | None = None,
    only_success: bool = True,
    image_writer_processes: int = 5,
    image_writer_threads: int = 10,
):
    """Convert one or more SoftMimicGen Franka towel HDF5 files to one LeRobot dataset.

    Args:
        hdf5: Path(s) to the SoftMimicGen HDF5 file(s).
        repo_name: Dataset identifier. If --output-dir isn't given, the dataset is written to
            $HF_LEROBOT_HOME/<repo_name>.
        output_dir: Write the dataset directly to this directory (recommended) -- pass the SAME directory
            as `local_root` to compute_norm_stats.py / train.py. Deleted first if it exists.
        prompt: Language instruction stored as every frame's task.
        state: State layout, "eef" or "joint" (see module docstring).
        image_size: Images are resized (with padding) to image_size x image_size. The model resizes to
            224 anyway; storing the native 512x512 only makes the dataset larger.
        max_episodes: If set, only convert the first N demos of each file (quick smoke test).
        only_success: Skip demos whose `success` attr is false.
        image_writer_processes / image_writer_threads: LeRobot's PNG-encoding workers.
    """
    hdf5_paths = [Path(p) for p in hdf5]
    output_path = Path(output_dir) if output_dir else HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Every file must run at the same control rate (and come from the same env).
    env_args = None
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            args = json.loads(f["data"].attrs["env_args"])
        if env_args is not None and (args.get("env_name"), args.get("sim_args")) != (
            env_args.get("env_name"),
            env_args.get("sim_args"),
        ):
            raise ValueError(f"{hdf5_path}: env_args differ from {hdf5_paths[0]}")
        env_args = args
    sim = env_args.get("sim_args", {})
    fps = round(1.0 / (sim.get("dt", 0.01) * sim.get("decimation", 5)))

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="panda",
        fps=fps,
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
            "state": {"dtype": "float32", "shape": (9,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    episodes: list[str] = []  # "<file>:<demo>" in dataset episode order
    num_frames = 0
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            if only_success:
                demo_keys = [k for k in demo_keys if bool(data[k].attrs.get("success", True))]
            demo_keys = demo_keys[:max_episodes]

            for demo_key in tqdm.tqdm(demo_keys, desc=hdf5_path.stem):
                demo = data[demo_key]
                actions = demo["actions"][()].astype(np.float32)
                states = _state(demo, state)
                room = image_tools.resize_with_pad(demo["obs"][ROOM_CAMERA][()], image_size, image_size)
                wrist = image_tools.resize_with_pad(demo["obs"][WRIST_CAMERA][()], image_size, image_size)

                for i in range(actions.shape[0]):
                    dataset.add_frame(
                        {
                            "image": room[i],
                            "wrist_image": wrist[i],
                            "state": states[i],
                            "actions": actions[i],
                            "task": prompt,
                        }
                    )
                dataset.save_episode()
                num_frames += actions.shape[0]
                episodes.append(f"{hdf5_path.name}:{demo_key}")

    info = {
        "source_hdf5": [str(p.resolve()) for p in hdf5_paths],
        "env_name": env_args.get("env_name"),
        "fps": fps,
        "episodes": episodes,
        "frames": num_frames,
        "prompt": prompt,
        "state_layout": state,
        "state_names": (
            ["eef_x", "eef_y", "eef_z", "quat_w", "quat_x", "quat_y", "quat_z", "finger_1", "finger_2"]
            if state == "eef"
            else [f"joint_{j}" for j in range(1, 8)] + ["finger_1", "finger_2"]
        ),
        "quat_canonicalization": "sign chosen so quat_x >= 0" if state == "eef" else None,
        "action_names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper(+1 open,-1 close)"],
        "image_size": image_size,
        "images_flipped": False,
    }
    (output_path / "meta" / "conversion_info.json").write_text(json.dumps(info, indent=1))
    print(f"Wrote {len(episodes)} episodes, {num_frames} frames to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
