"""Run one LIBERO episode and visualize, per inferred action chunk, which words of
the language prompt the action-expert attended to most -- as a heatmap displayed
side-by-side with the rollout video.

This is a diagnostic/visualization tool, not the standard eval path
(examples/libero/main.py + scripts/serve_policy.py). Like main.py, it talks to a
policy server over a websocket -- it does NOT import openpi/JAX directly, since
LIBERO's environment and the full openpi model package have hard-conflicting
pinned dependencies (LIBERO needs Python 3.8 + torch==1.11.0+cu113; openpi needs
Python >=3.11 + torch==2.7.1) and cannot be installed together. Run this in the
same LIBERO environment as main.py (see examples/libero/README.md), pointed at a
server started with scripts/serve_policy_with_attention.py (not the regular
serve_policy.py -- that one doesn't return the extra `attn`/`prefix_len` fields
this script needs).

Usage:
    # Terminal 1:
    uv run scripts/serve_policy_with_attention.py policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

    # Terminal 2 (in the LIBERO env):
    python examples/libero/visualize_attention.py \
        --task-suite-name libero_10 --task-id 0
"""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
import urllib.request

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import sentencepiece
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Same file openpi.models.tokenizer.PaligemmaTokenizer downloads server-side (via
# GCS auth there; this is the public, anonymous-access HTTPS mirror of the same
# object, so the client doesn't need any GCS/openpi dependency to fetch it).
TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_10"
    task_id: int = 0
    episode_idx: int = 0
    num_steps_wait: int = 10
    max_steps: int = 520

    # If set, overrides the LIBERO task's own built-in language instruction with
    # this string -- sent to the model as the prompt instead, and used for the
    # client's own re-tokenization (so the two stay aligned). The environment
    # itself (physics, success condition) is unaffected -- it's driven by the
    # task's .bddl file, not by whatever string the model is fed -- so this is
    # a clean way to test whether the model's attention tracks word content or
    # word position: same scene, same dynamics, differently-worded instruction.
    prompt_override: str | None = None

    # Must match the served model's tokenizer max length (48 for the standard
    # pi0/pi05 configs) -- the client re-tokenizes the prompt itself to recover
    # word boundaries, so it needs to pad/mask identically to the server.
    max_token_len: int = 48

    seed: int = 7
    video_out_path: str = "data/libero/attention_videos"
    tokenizer_cache_dir: str = "data/libero/.tokenizer_cache"

    # Alpha-blend strength for the image-attention heatmap overlays (0=invisible,
    # 1=heatmap only, no underlying camera frame).
    image_attn_alpha: float = 0.45


def _get_libero_env(task, resolution, seed, horizon=1000):
    """Copied from examples/libero/main.py -- see that file for context."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
        horizon=horizon,
    )
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Copied from examples/libero/main.py (originally from robosuite)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def load_tokenizer(cache_dir: str) -> sentencepiece.SentencePieceProcessor:
    cache_path = pathlib.Path(cache_dir) / "paligemma_tokenizer.model"
    if not cache_path.is_file():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading tokenizer to {cache_path}...")
        urllib.request.urlretrieve(TOKENIZER_URL, cache_path)  # noqa: S310
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(str(cache_path))
    return sp


def tokenize_prompt(
    tokenizer: sentencepiece.SentencePieceProcessor, prompt: str, max_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduces openpi.models.tokenizer.PaligemmaTokenizer.tokenize()'s exact
    convention, so the token ids/mask line up with what the server used for the
    `attn` array's language columns.
    """
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    tokens = tokenizer.encode(cleaned_text, add_bos=True) + tokenizer.encode("\n")
    tokens_len = len(tokens)
    if tokens_len < max_len:
        token_ids = np.array(tokens + [0] * (max_len - tokens_len))
        token_mask = np.array([True] * tokens_len + [False] * (max_len - tokens_len))
    else:
        token_ids = np.array(tokens[:max_len])
        token_mask = np.array([True] * max_len)
    return token_ids, token_mask


def group_tokens_into_words(
    tokenizer: sentencepiece.SentencePieceProcessor, token_ids: np.ndarray, token_mask: np.ndarray
) -> list[tuple[str, list[int]]]:
    """Groups a PaligemmaTokenizer-encoded prompt's token ids into words.

    Returns a list of (word_text, [column_indices_into_token_ids]) pairs, in
    order, skipping the BOS token, padding, and the trailing "\\n" (added by
    PaligemmaTokenizer.tokenize as the "start of answer" marker, not a prompt
    word). SentencePiece marks the start of each real word with a "▁" (U+2581)
    prefix on its first piece; consecutive pieces without that prefix continue
    the same word.
    """
    words: list[tuple[str, list[int]]] = []
    current_pieces: list[str] = []
    current_cols: list[int] = []

    def flush():
        if current_pieces:
            text = "".join(current_pieces).replace("▁", " ").strip()
            if text and text != "\n":
                words.append((text, list(current_cols)))
        current_pieces.clear()
        current_cols.clear()

    # Note: no strict=True here (added in Python 3.10) -- this script runs in
    # LIBERO's Python 3.8 environment. token_ids/token_mask are always
    # constructed with matching length, so this is safe without it.
    for col, (token_id, valid) in enumerate(zip(token_ids, token_mask)):
        if not valid:
            continue
        piece = tokenizer.IdToPiece(int(token_id))
        if piece in ("<bos>", "<eos>", "<pad>", "\n"):
            # "\n" is skipped rather than just filtered by the `flush()` text check
            # below, since without a leading "▁" it would otherwise silently merge
            # into whichever word came right before it (it's PaligemmaTokenizer's
            # separately-appended "start of answer" marker, not a prompt word).
            continue
        if piece.startswith("▁") or not current_pieces:
            flush()
        current_pieces.append(piece)
        current_cols.append(col)
    flush()
    return words


def render_attention_heatmap(
    word_attn: np.ndarray,
    words: list[str],
    row_labels: list[str],
    size: tuple[int, int],
    lang_mass_pct: float | None = None,
) -> np.ndarray:
    """Renders a [num_rows, num_words] attention matrix as an RGB image array.

    `word_attn` holds each word's *raw* softmax weight (share of the model's
    full attention over images + language + suffix, typically 800+ positions
    for pi05_libero -- ~768 image patch tokens alone), so even genuinely
    selective attention lands around 0.001-0.02: too small to show any visible
    contrast on a straight vmin=0/vmax=max scale, since that scale is set by
    whatever the single largest raw value happens to be, and everything else
    is compared against that -- not because attention is actually uniform.
    This renormalizes each row to the words' *relative* share of each other
    (so the row sums to 1 across just the words shown, ignoring images/self-
    attention), which reveals genuine relative preference regardless of how
    small the absolute values are. `lang_mass_pct`, if given, is displayed
    separately so you can still tell whether the model attends to language
    much at all in absolute terms.

    `size` is (width, height) in pixels, matched to the rollout frame size so
    the two can be concatenated side-by-side into one video.
    """
    row_sums = word_attn.sum(axis=-1, keepdims=True)
    word_attn_rel = word_attn / np.clip(row_sums, 1e-12, None)

    width_px, height_px = size
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    im = ax.imshow(word_attn_rel, aspect="auto", cmap="viridis", vmin=0.0, vmax=word_attn_rel.max())
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    title = "Attention over prompt words (relative)"
    if lang_mass_pct is not None:
        title += f"\n(language attn mass: {lang_mass_pct:.2f}% of total)"
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def render_rank_heatmap(
    word_attn: np.ndarray, words: list[str], row_labels: list[str], size: tuple[int, int]
) -> np.ndarray:
    """Renders which word ranks highest-to-lowest by attention, per row.

    Complements render_attention_heatmap: ranks always span the full 1..N
    range regardless of how close together (or how small in absolute terms)
    the underlying raw attention values are, so this never washes out the way
    a value-based heatmap can when one row's peak dwarfs the others' (their
    own, smaller-but-real internal structure gets crushed toward the bottom of
    whatever shared color scale that peak sets). The tradeoff: it shows
    *ordering* only, not magnitude or confidence -- a near-tie for 1st place
    looks identical here to a landslide.
    """
    num_words = word_attn.shape[-1]
    order = np.argsort(-word_attn, axis=-1)
    ranks = np.empty_like(order)
    row_idx = np.arange(word_attn.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, num_words + 1)[None, :]

    width_px, height_px = size
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    im = ax.imshow(ranks, aspect="auto", cmap="viridis_r", vmin=1, vmax=num_words)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Attention RANK (brightest = rank 1 = highest)", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("rank (1=highest)", fontsize=7)
    fig.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def render_image_attention_overlay(
    frame_rgb: np.ndarray, attn_grid: np.ndarray, alpha: float = 0.45, cmap_name: str = "inferno"
) -> np.ndarray:
    """Alpha-blends a per-patch attention grid as a Grad-CAM-style heatmap over a
    camera frame.

    `attn_grid` is a 2D [grid_h, grid_w] array of raw attention weight for one
    camera's patch tokens (already averaged over layers, heads, and the action
    chunk's suffix steps by the caller). Like the language heatmaps, the raw
    values are a tiny slice of an 800+-position softmax, so this renormalizes
    to this one frame's own min/max to reveal whatever relative spatial
    structure exists, rather than rendering as a flat color.

    Runs on every rollout frame (not just once per action chunk), so unlike
    render_attention_heatmap/render_rank_heatmap this deliberately avoids
    matplotlib figure/canvas machinery -- it's a handful of vectorized array
    ops instead, which is orders of magnitude cheaper per call.
    """
    from PIL import Image

    grid = attn_grid.astype(np.float32)
    grid = grid - grid.min()
    peak = grid.max()
    if peak > 1e-12:
        grid = grid / peak

    cmap = matplotlib.colormaps[cmap_name]
    heat_rgb = (cmap(grid)[..., :3] * 255).astype(np.uint8)
    h, w = frame_rgb.shape[:2]
    heat_rgb_resized = np.asarray(Image.fromarray(heat_rgb).resize((w, h), Image.BILINEAR)).astype(np.float32)

    blended = (1.0 - alpha) * frame_rgb.astype(np.float32) + alpha * heat_rgb_resized
    return np.clip(blended, 0, 255).astype(np.uint8)


def _hstack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Resizes all frames to the first frame's height and concatenates them side by side."""
    from PIL import Image

    target_h = frames[0].shape[0]
    resized = []
    for f in frames:
        if f.shape[0] != target_h:
            scale = target_h / f.shape[0]
            new_w = max(1, int(round(f.shape[1] * scale)))
            f = np.asarray(Image.fromarray(f).resize((new_w, target_h)))
        resized.append(f)
    return np.concatenate(resized, axis=1)


def main(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    sp_tokenizer = load_tokenizer(args.tokenizer_cache_dir)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    env, task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, args.seed, horizon=args.max_steps + args.num_steps_wait
    )
    if args.prompt_override is not None:
        # Same underscore-for-space convention the tokenizer itself uses (see
        # openpi.models.tokenizer) -- lets the override survive being passed
        # through an unquoted shell variable (e.g. Docker's CLIENT_ARGS) without
        # word-splitting on embedded spaces.
        override = args.prompt_override.replace("_", " ").strip()
        logging.info(f"Overriding task instruction {task_description!r} -> {override!r}")
        task_description = override
    logging.info(f"Task: {task_description}")

    logging.info("Resetting env...")
    env.reset()
    logging.info("Setting initial state...")
    obs = env.set_init_state(initial_states[args.episode_idx])
    logging.info("Env ready.")

    action_plan: list[np.ndarray] = []
    replay_images = []
    value_heatmap_frames = []
    rank_heatmap_frames = []
    base_overlay_frames = []
    wrist_overlay_frames = []
    t = 0
    done = False
    value_heatmap_frame = None
    rank_heatmap_frame = None
    base_attn_grid = None
    wrist_attn_grid = None
    warned_no_camera_fields = False

    # Tokenize once -- the prompt doesn't change across the episode.
    token_ids, token_mask = tokenize_prompt(sp_tokenizer, str(task_description), args.max_token_len)
    words_with_cols = group_tokens_into_words(sp_tokenizer, token_ids, token_mask)
    lang_len = token_ids.shape[0]
    logging.info(f"Prompt tokenized into {len(words_with_cols)} words; starting rollout "
                 f"({args.num_steps_wait} warmup steps, then up to {args.max_steps} real steps)...")

    while t < args.max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                if t == 0:
                    logging.info("Running warmup steps...")
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            replay_images.append(img)

            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                logging.info(f"Requesting inference from server (step {t})...")
                result = client.infer(element)
                logging.info("Got action chunk back from server.")
                action_chunk = result["actions"]
                assert len(action_chunk) >= args.replan_steps, (
                    f"We want to replan every {args.replan_steps} steps, "
                    f"but policy only predicts {len(action_chunk)} steps."
                )
                action_plan.extend(action_chunk[: args.replan_steps])

                # --- Build the attention heatmaps for this action chunk ---
                # Rows = every transformer layer (heads already averaged away
                # server-side, but never layers -- see AttentionCapturingPolicy).
                # Averaged over the action-chunk's suffix_len steps instead: within
                # one chunk, every predicted step shares the same prompt and
                # observation, so which *layer* attends to which word is the more
                # informative axis to show than which step does.
                attn_all_layers = result["attn"]  # [depth, suffix_len, prefix_len + suffix_len]
                prefix_len = result["prefix_len"]
                attn_lang = attn_all_layers[:, :, prefix_len - lang_len : prefix_len]  # [depth, suffix_len, lang_len]
                attn_lang_per_layer = attn_lang.mean(axis=1)  # -> [depth, lang_len]

                if words_with_cols:
                    words = [w for w, _ in words_with_cols]
                    word_attn = np.stack(
                        [attn_lang_per_layer[:, cols].mean(axis=-1) for _, cols in words_with_cols], axis=-1
                    )  # [depth, num_words]
                else:
                    words = ["(no prompt words)"]
                    word_attn = np.zeros((attn_lang_per_layer.shape[0], 1), dtype=np.float32)

                # Total raw attention mass on language tokens (not renormalized) --
                # separate from word_attn's per-word *relative* share, this tells
                # you whether the model attends to language much at all in
                # absolute terms (see render_attention_heatmap's docstring).
                lang_mass_pct = float(attn_lang_per_layer.sum(axis=-1).mean()) * 100

                row_labels = [f"layer{i}" for i in range(word_attn.shape[0])]
                value_heatmap_frame = render_attention_heatmap(
                    word_attn, words, row_labels, size=(args.resize_size, args.resize_size),
                    lang_mass_pct=lang_mass_pct,
                )
                rank_heatmap_frame = render_rank_heatmap(
                    word_attn, words, row_labels, size=(args.resize_size, args.resize_size)
                )

                # --- Build the image-attention grids for this action chunk ---
                # Unlike the language heatmaps, this DOES average over layers --
                # there's no single 2D "words" axis to keep separate rows for
                # here, and the goal is a qualitative "does the hotspot track the
                # object across the rollout", not a per-layer breakdown (which
                # would need one overlay panel per layer to show honestly).
                camera_names = result.get("camera_names")
                patches_per_camera = result.get("patches_per_camera")
                if camera_names is None or patches_per_camera is None:
                    if not warned_no_camera_fields:
                        logging.warning(
                            "Server response has no camera_names/patches_per_camera -- "
                            "restart the server from the version of attention_policy.py "
                            "with image-attention support to see the overlay panels."
                        )
                        warned_no_camera_fields = True
                    base_attn_grid = None
                    wrist_attn_grid = None
                else:
                    grid_side = int(round(math.sqrt(patches_per_camera)))
                    assert grid_side * grid_side == patches_per_camera, (
                        f"patches_per_camera={patches_per_camera} is not a perfect square "
                        f"(got side {grid_side})"
                    )
                    attn_img_all = attn_all_layers[:, :, :prefix_len - lang_len]  # [depth, suffix_len, num_img_cols]

                    def _camera_grid(camera_name: str) -> np.ndarray | None:
                        if camera_name not in camera_names:
                            return None
                        cam_idx = camera_names.index(camera_name)
                        start = cam_idx * patches_per_camera
                        cam_attn = attn_img_all[:, :, start : start + patches_per_camera]  # [depth, suffix_len, P]
                        return cam_attn.mean(axis=(0, 1)).reshape(grid_side, grid_side)

                    base_attn_grid = _camera_grid("base_0_rgb")
                    wrist_attn_grid = _camera_grid("left_wrist_0_rgb")

            action = action_plan.pop(0)
            value_heatmap_frames.append(value_heatmap_frame)
            rank_heatmap_frames.append(rank_heatmap_frame)
            base_overlay_frames.append(
                render_image_attention_overlay(img, base_attn_grid, alpha=args.image_attn_alpha)
                if base_attn_grid is not None
                else img
            )
            wrist_overlay_frames.append(
                render_image_attention_overlay(wrist_img, wrist_attn_grid, alpha=args.image_attn_alpha)
                if wrist_attn_grid is not None
                else wrist_img
            )

            obs, reward, done, info = env.step(action.tolist())
            if done:
                break
            t += 1

        except Exception as e:  # noqa: BLE001
            logging.error(f"Caught exception: {e}")
            break

    suffix = "success" if done else "failure"
    task_segment = task_description.replace(" ", "_")[:80]
    out_path = pathlib.Path(args.video_out_path) / f"attn_{task_segment}_ep{args.episode_idx}_{suffix}.mp4"

    # No strict=True (Python 3.10+) -- see note in group_tokens_into_words.
    # Panel order: plain rollout | base-camera attention overlay | wrist-camera
    # attention overlay | word-value heatmap | word-rank heatmap.
    frames = [
        _hstack_frames([r, bo, wo, v, k])
        for r, bo, wo, v, k in zip(
            replay_images, base_overlay_frames, wrist_overlay_frames, value_heatmap_frames, rank_heatmap_frames
        )
    ]
    imageio.mimwrite(out_path, frames, fps=10)
    logging.info(f"Wrote {out_path} ({len(frames)} frames). Success: {done}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
