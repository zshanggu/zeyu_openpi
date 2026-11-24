"""Copy a SoftMimicGen Franka towel HDF5 file, replacing its camera images with frames of Cosmos videos.

Input:
  * --hdf5: the source file, e.g. /data/hyeonahj/share/craft_hdf5/franka_towel_n750_seeds23.hdf5
    (data/demo_N/obs/{agentview_image, robot0_eye_in_hand_image}, each (T, 512, 512, 3) uint8).
  * --videos: a folder of replacement videos, e.g.
    /data/hyeonahj/share/franka_towel_n750_seeds1234_videos/videos/O, searched recursively for
    demo_NNN_<suffix>.mp4 (default suffix 01). Each video is 1024x512 with one frame per HDF5 step:
    the left half is the room camera (-> agentview_image), the right half the wrist camera
    (-> robot0_eye_in_hand_image).

  * --demos (optional): a text file listing the demos to keep, one per line as demo_007, demo_7 or 7
    (blank lines and # comments ignored), e.g. a list exported by the CoTracker GUI batch:
    zeyu-co-tracker/outputs/gui/<batch>/demos_overall_mean_top80_vlmscore_lt0.2.txt. Only these demos
    get their images replaced and are written; every listed demo must exist in the HDF5.

Video demo_NNN belongs to HDF5 demo_N (demo_001 -> demo_1; checked against demo_001_source_all.mp4,
the simulator video the Cosmos run was conditioned on).

Output: the same file layout, with only the demos that have a replacement video (and, with --demos,
are listed). Everything else is
copied unchanged (actions, states, attrs, and also the simulator geoedge / instance images, which
still describe the simulator frames). A demo with no video is left out of the output, and so is a demo
whose video's frame count or frame size does not match the HDF5 (printed with the reason). Demo names
are kept (no renumbering),
and data.attrs["total"] is the number of frames written. Each written demo gets the attrs
image_source_video (the video path) and image_source_hdf5; with --demos, data.attrs["image_demo_list"]
is the list file.

With --keep-unreplaced, every source demo that is not replaced (not listed in --demos, no video, a
mismatching video, or beyond --max-episodes) is still written, copied unchanged with its simulator
images, so the output has as many demos as the source (e.g. 40 listed of 50 -> 40 replaced + 10 kept).
Every written demo then gets attrs image_replaced (True/False); kept demos have no image_source_video.

Usage:
    uv run examples/franka_towel/replace_towel_hdf5_images.py \
        --hdf5 /data/hyeonahj/share/craft_hdf5/franka_towel_n750_seeds23.hdf5 \
        --videos /data/hyeonahj/share/franka_towel_n750_seeds1234_videos/videos/O \
        --out /data/zshanggu/Deformable/hdf5/franka_towel_n750_seeds23_cosmos_O.hdf5

    # only the demos listed in a text file
    uv run examples/franka_towel/replace_towel_hdf5_images.py \
        --hdf5 franka_towel_zeyu_n50_seeds23.hdf5 \
        --videos /home/zeyu/PHD_LAB/DEFORMABLE/hdf5/simulation_from_eric/franka_towel_zeyu_n50_seeds23 \
        --demos .../outputs/gui/franka_towel_zeyu_n50_seeds23/demos_overall_mean_top80_vlmscore_lt0.2.txt \
        --out franka_towel_zeyu_n50_seeds23_cosmos_selected.hdf5

    # the listed demos replaced, all other demos kept with their simulator images
    uv run examples/franka_towel/replace_towel_hdf5_images.py ... --demos <list>.txt --keep-unreplaced \
        --out franka_towel_zeyu_n50_seeds23_cosmos_selected_plus_sim.hdf5

Speed: --workers processes (default 8) decode the videos and gzip the new image chunks in parallel; the main
process writes the compressed chunks as they are (h5py write_direct_chunk), in demo order, so the HDF5 is still
written by one process. The output is the same as writing the images with h5py's own gzip filter. A source image
dataset with other filters (shuffle, fletcher32, not gzip) falls back to plain writes in the main process.
--workers 0 does everything in the main process.

Then convert it like any other towel file with convert_towel_hdf5_to_lerobot.py.
"""

from __future__ import annotations

import itertools
import multiprocessing
from pathlib import Path
import re
import zlib

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


def read_demo_list(path: Path) -> list[str]:
    """HDF5 demo keys ("demo_7") listed in a text file, one per line as demo_007, demo_7 or 7, in file
    order without repeats. Blank lines and # comments are ignored."""
    keys: list[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        m = re.fullmatch(r"(?:demo_)?(\d+)", text)
        if not m:
            raise ValueError(f"{path}:{line_no}: not a demo name: {line!r}")
        key = f"demo_{int(m.group(1))}"
        if key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError(f"{path}: no demos listed")
    return keys


def read_video(path: Path) -> np.ndarray:
    """All frames as (T, H, W, 3) uint8 RGB."""
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def chunk_layout(src: h5py.Dataset) -> dict | None:
    """How a worker can compress data for `src` itself: its chunk shape and gzip level (None: no compression).
    None when the dataset uses anything write_direct_chunk + zlib can't reproduce (then the main process
    writes it with h5py's filters)."""
    if src.chunks is None or src.shuffle or src.fletcher32 or src.scaleoffset is not None:
        return None
    if src.compression not in (None, "gzip"):
        return None
    return {"chunks": src.chunks, "level": src.compression_opts if src.compression == "gzip" else None}


def compress_chunks(data: np.ndarray, layout: dict) -> list[tuple[tuple[int, ...], bytes]]:
    """(chunk offset, stored bytes) for every chunk of `data`, as HDF5's deflate filter stores them (a zlib
    stream). Edge chunks are padded to the full chunk shape with the fill value 0, as HDF5 stores them."""
    chunks, level = layout["chunks"], layout["level"]
    out = []
    for offset in itertools.product(*(range(0, n, c) for n, c in zip(data.shape, chunks))):
        block = data[tuple(slice(o, o + c) for o, c in zip(offset, chunks))]
        if block.shape != tuple(chunks):
            block = np.pad(block, [(0, c - b) for c, b in zip(chunks, block.shape)])
        raw = np.ascontiguousarray(block).tobytes()
        out.append((offset, raw if level is None else zlib.compress(raw, level)))
    return out


def prepare_demo(job: dict) -> dict:
    """Worker: decode one video, check it against the demo, split it into room | wrist and compress their chunks
    (or return the plain arrays when a layout is None). Touches no HDF5 file."""
    frames = read_video(job["video"])
    num_steps, height, width = job["shape"]
    if len(frames) != num_steps:
        return {"skip": f"video has {len(frames)} frames, hdf5 has {num_steps}"}
    if frames.shape[1:3] != (height, 2 * width) or job["wrist_hw"] != (height, width):
        return {"skip": f"video frame {frames.shape[2]}x{frames.shape[1]}, expected {2 * width}x{height}"}
    out = {}
    for name, data in ((ROOM_CAMERA, frames[:, :, :width]), (WRIST_CAMERA, frames[:, :, width:])):
        data = data.astype(job["dtype"][name], copy=False)
        layout = job["layout"][name]
        out[name] = ("chunks", compress_chunks(data, layout)) if layout else ("array", np.ascontiguousarray(data))
    return out


def write_prepared(src: h5py.Dataset, dst: h5py.Group, prepared: tuple) -> None:
    """Create a dataset like `src` in `dst` and fill it with what prepare_demo returned."""
    kind, payload = prepared
    if kind == "array":
        copy_like(src, dst, payload)
        return
    ds = dst.create_dataset(
        Path(src.name).name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=src.chunks,
        compression=src.compression,
        compression_opts=src.compression_opts,
    )
    for offset, data in payload:
        ds.id.write_direct_chunk(offset, data)


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
    demos: str | None = None,
    max_episodes: int | None = None,
    keep_unreplaced: bool = False,
    workers: int = 8,
    overwrite: bool = False,
) -> None:
    """
    Args:
        hdf5: Source SoftMimicGen HDF5 file.
        videos: Folder searched recursively for demo_NNN_<suffix>.mp4.
        out: Output HDF5 file.
        suffix: Video name suffix: demo_NNN_<suffix>.mp4.
        demos: Text file listing the demos to keep (demo_007, demo_7 or 7 per line); only these are
            written. Default: every demo that has a video.
        max_episodes: If set, only replace the first N matched demos (quick test).
        keep_unreplaced: Also write every source demo that is not replaced, unchanged (simulator
            images), so the output keeps all source demos.
        workers: Processes that decode videos and compress images in parallel (0: all in the main process).
        overwrite: Replace `out` if it exists.
    """
    hdf5_path, videos_path, out_path = Path(hdf5), Path(videos), Path(out)
    if out_path.resolve() == hdf5_path.resolve():
        raise ValueError("--out must differ from --hdf5")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (pass --overwrite)")
    listed = read_demo_list(Path(demos)) if demos is not None else None
    if listed is not None:
        print(f"{len(listed)} demos listed in {demos}")
        with h5py.File(hdf5_path, "r") as src:  # check before any output is created
            missing = [k for k in listed if k not in src["data"]]
        if missing:  # most likely a list made for another dataset
            raise ValueError(f"{len(missing)} listed demos are not in {hdf5_path}: {missing}")
    video_of = find_videos(videos_path, suffix)
    print(f"{len(video_of)} videos demo_*_{suffix}.mp4 in {videos_path}")

    # Started before any HDF5 file is open, so no worker inherits an open h5py handle.
    pool = multiprocessing.get_context("fork").Pool(workers) if workers > 0 else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".partial")
    skipped: dict[str, str] = {}
    written: list[str] = []
    kept: list[str] = []
    total = 0
    with h5py.File(hdf5_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        src_data = src["data"]
        dst_data = dst.create_group("data")
        for k, v in src_data.attrs.items():
            dst_data.attrs[k] = v

        demo_keys = sorted(src_data.keys(), key=lambda k: int(k.split("_")[1]))
        all_keys = list(demo_keys)
        unused = sorted(set(video_of) - set(demo_keys))
        if unused:
            print(f"warning: {len(unused)} videos have no demo in the HDF5: {unused}")
        not_listed = 0
        if listed is not None:
            dst_data.attrs["image_demo_list"] = str(Path(demos).resolve())
            not_listed = len(demo_keys) - len(listed)
            demo_keys = [k for k in demo_keys if k in set(listed)]
        candidates = [k for k in demo_keys if k in video_of]
        for k in demo_keys:
            if k not in video_of:
                skipped[k] = "no video"
        if max_episodes is not None:
            candidates = candidates[:max_episodes]

        jobs = []
        for key in candidates:
            obs = src_data[key]["obs"]
            room, wrist = obs[ROOM_CAMERA], obs[WRIST_CAMERA]
            jobs.append({
                "video": video_of[key],
                "shape": room.shape[:3],
                "wrist_hw": wrist.shape[1:3],
                "dtype": {ROOM_CAMERA: room.dtype, WRIST_CAMERA: wrist.dtype},
                "layout": {ROOM_CAMERA: chunk_layout(room), WRIST_CAMERA: chunk_layout(wrist)},
            })
        # Results come back in demo order; workers run ahead while the main process writes.
        results = pool.imap(prepare_demo, jobs) if pool else map(prepare_demo, jobs)

        for key, prepared in tqdm.tqdm(zip(candidates, results), total=len(candidates), desc=out_path.stem):
            demo = src_data[key]
            room, wrist = demo["obs"][ROOM_CAMERA], demo["obs"][WRIST_CAMERA]
            num_steps = room.shape[0]
            if "skip" in prepared:
                skipped[key] = prepared["skip"]
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
            write_prepared(room, out_obs, prepared[ROOM_CAMERA])
            write_prepared(wrist, out_obs, prepared[WRIST_CAMERA])
            del prepared
            out_demo.attrs["image_source_video"] = str(video_of[key])
            out_demo.attrs["image_source_hdf5"] = str(hdf5_path)
            if keep_unreplaced:
                out_demo.attrs["image_replaced"] = True
            written.append(key)
            total += num_steps

        if keep_unreplaced:
            dst_data.attrs["keep_unreplaced"] = True
            for key in tqdm.tqdm([k for k in all_keys if k not in set(written)], desc="keep unreplaced"):
                src.copy(src_data[key], dst_data, name=key)  # as stored, simulator images
                dst_data[key].attrs["image_replaced"] = False
                dst_data[key].attrs["image_source_hdf5"] = str(hdf5_path)
                kept.append(key)
                total += src_data[key]["obs"][ROOM_CAMERA].shape[0]

        dst_data.attrs["total"] = str(total) if isinstance(src_data.attrs.get("total"), str) else total
    if pool:
        pool.close()
        pool.join()
    tmp_path.replace(out_path)

    reasons: dict[str, list[str]] = {}
    for key, reason in skipped.items():
        reasons.setdefault("no video" if reason == "no video" else "mismatch", []).append(f"{key}: {reason}")
    print(f"Wrote {len(written) + len(kept)} demos, {total} frames to {out_path}")
    print(f"  {len(written)} with replaced images" + (f", {len(kept)} kept unchanged (simulator images)"
                                                     if keep_unreplaced else ""))
    if keep_unreplaced:
        return  # nothing was left out
    if listed is not None:
        print(f"Left out {not_listed} source demos not listed in {demos}")
        for line in reasons.get("no video", []):  # a listed demo without its video is worth a line each
            print(f"Left out listed {line}")
    else:
        print(f"Left out {len(reasons.get('no video', []))} source demos that have no video")
    for line in reasons.get("mismatch", []):
        print(f"Left out {line}")


if __name__ == "__main__":
    tyro.cli(main)
