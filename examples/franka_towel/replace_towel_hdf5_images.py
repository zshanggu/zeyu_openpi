"""Copy a SoftMimicGen Franka towel HDF5 file, replacing its camera images with frames of Cosmos videos.

Input:
  * --hdf5: the source file, e.g. /data/hyeonahj/share/craft_hdf5/franka_towel_n750_seeds23.hdf5
    (data/demo_N/obs/{agentview_image, robot0_eye_in_hand_image}, each (T, 512, 512, 3) uint8).
  * --videos: a folder of replacement videos, e.g.
    /data/hyeonahj/share/franka_towel_n750_seeds1234_videos/videos/O, searched recursively for
    demo_NNN_<suffix>.mp4 (default suffix 01). Each video is 1024x512 with one frame per HDF5 step:
    the left half is the room camera (-> agentview_image), the right half the wrist camera
    (-> robot0_eye_in_hand_image).

Video demo_NNN belongs to HDF5 demo_N (demo_001 -> demo_1; checked against demo_001_source_all.mp4,
the simulator video the Cosmos run was conditioned on).

Output: the same file layout, with only the demos that have a replacement video. Everything else is
copied unchanged (actions, states, attrs, and also the simulator geoedge / instance images, which
still describe the simulator frames). A demo with no video is left out of the output, and so is a demo
whose video's frame count or frame size does not match the HDF5 (printed with the reason). Demo names
are kept (no renumbering),
and data.attrs["total"] is the number of frames written. Each written demo gets the attrs
image_source_video (the video path) and image_source_hdf5.

Usage:
    uv run examples/franka_towel/replace_towel_hdf5_images.py \
        --hdf5 /data/hyeonahj/share/craft_hdf5/franka_towel_n750_seeds23.hdf5 \
        --videos /data/hyeonahj/share/franka_towel_n750_seeds1234_videos/videos/O \
        --out /data/zshanggu/Deformable/hdf5/franka_towel_n750_seeds23_cosmos_O.hdf5

Then convert it like any other towel file with convert_towel_hdf5_to_lerobot.py.
"""

from __future__ import annotations

from pathlib import Path
import re

import av
import h5py
import numpy as np
import tqdm
import tyro

ROOM_CAMERA = "agentview_image"
WRIST_CAMERA = "robot0_eye_in_hand_image"


def find_videos(videos: Path, suffix: str) -> dict[str, Path]:
    """HDF5 demo key ("demo_1") -> path of its demo_001_<suffix>.mp4."""
    pattern = re.compile(rf"demo_(\d+)_{re.escape(suffix)}\.mp4")
    found: dict[str, Path] = {}
    for path in sorted(videos.rglob(f"demo_*_{suffix}.mp4")):
        m = pattern.fullmatch(path.name)
        if not m:
            continue
        key = f"demo_{int(m.group(1))}"
        if key in found:
            raise ValueError(f"two videos for {key}: {found[key]} and {path}")
        found[key] = path
    return found


def read_video(path: Path) -> np.ndarray:
    """All frames as (T, H, W, 3) uint8 RGB."""
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def copy_like(src: h5py.Dataset, dst: h5py.Group, data: np.ndarray) -> None:
    """Write `data` under src's name with src's dtype, chunking and compression."""
    dst.create_dataset(
        Path(src.name).name,
        data=data.astype(src.dtype, copy=False),
        chunks=src.chunks,
        compression=src.compression,
        compression_opts=src.compression_opts,
        shuffle=src.shuffle,
        fletcher32=src.fletcher32,
    )


def main(
    hdf5: str,
    videos: str,
    out: str,
    *,
    suffix: str = "01",
    max_episodes: int | None = None,
    overwrite: bool = False,
) -> None:
    """
    Args:
        hdf5: Source SoftMimicGen HDF5 file.
        videos: Folder searched recursively for demo_NNN_<suffix>.mp4.
        out: Output HDF5 file.
        suffix: Video name suffix: demo_NNN_<suffix>.mp4.
        max_episodes: If set, only write the first N matched demos (quick test).
        overwrite: Replace `out` if it exists.
    """
    hdf5_path, videos_path, out_path = Path(hdf5), Path(videos), Path(out)
    if out_path.resolve() == hdf5_path.resolve():
        raise ValueError("--out must differ from --hdf5")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (pass --overwrite)")
    video_of = find_videos(videos_path, suffix)
    print(f"{len(video_of)} videos demo_*_{suffix}.mp4 in {videos_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".partial")
    skipped: dict[str, str] = {}
    written: list[str] = []
    total = 0
    with h5py.File(hdf5_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        src_data = src["data"]
        dst_data = dst.create_group("data")
        for k, v in src_data.attrs.items():
            dst_data.attrs[k] = v

        demo_keys = sorted(src_data.keys(), key=lambda k: int(k.split("_")[1]))
        unused = sorted(set(video_of) - set(demo_keys))
        if unused:
            print(f"warning: {len(unused)} videos have no demo in the HDF5: {unused}")
        candidates = [k for k in demo_keys if k in video_of]
        for k in demo_keys:
            if k not in video_of:
                skipped[k] = "no video"
        if max_episodes is not None:
            candidates = candidates[:max_episodes]

        for key in tqdm.tqdm(candidates, desc=out_path.stem):
            demo = src_data[key]
            room, wrist = demo["obs"][ROOM_CAMERA], demo["obs"][WRIST_CAMERA]
            num_steps, height, width = room.shape[:3]
            frames = read_video(video_of[key])
            if len(frames) != num_steps:
                skipped[key] = f"video has {len(frames)} frames, hdf5 has {num_steps}"
                tqdm.tqdm.write(f"leave out {key}: {skipped[key]} ({video_of[key]})")
                continue
            if frames.shape[1:3] != (height, 2 * width) or wrist.shape[1:3] != (height, width):
                skipped[key] = f"video frame {frames.shape[2]}x{frames.shape[1]}, expected {2 * width}x{height}"
                tqdm.tqdm.write(f"leave out {key}: {skipped[key]} ({video_of[key]})")
                continue

            # Everything except the two camera images is copied as stored (no recompression).
            out_demo = dst_data.create_group(key)
            for name, obj in demo.attrs.items():
                out_demo.attrs[name] = obj
            for name in demo:
                if name != "obs":
                    src.copy(demo[name], out_demo, name=name)
            out_obs = out_demo.create_group("obs")
            for name, obj in demo["obs"].attrs.items():
                out_obs.attrs[name] = obj
            for name in demo["obs"]:
                if name not in (ROOM_CAMERA, WRIST_CAMERA):
                    src.copy(demo["obs"][name], out_obs, name=name)
            copy_like(room, out_obs, frames[:, :, :width])
            copy_like(wrist, out_obs, frames[:, :, width:])
            out_demo.attrs["image_source_video"] = str(video_of[key])
            out_demo.attrs["image_source_hdf5"] = str(hdf5_path)
            written.append(key)
            total += num_steps

        dst_data.attrs["total"] = str(total) if isinstance(src_data.attrs.get("total"), str) else total
    tmp_path.replace(out_path)

    reasons: dict[str, list[str]] = {}
    for key, reason in skipped.items():
        reasons.setdefault("no video" if reason == "no video" else "mismatch", []).append(f"{key}: {reason}")
    print(f"Wrote {len(written)} demos, {total} frames to {out_path}")
    print(f"Left out {len(reasons.get('no video', []))} source demos that have no video")
    for line in reasons.get("mismatch", []):
        print(f"Left out {line}")


if __name__ == "__main__":
    tyro.cli(main)
