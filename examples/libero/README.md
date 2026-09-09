# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match --build-constraint examples/libero/requirements.txt
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Converting LIBERO data to LeRobot format

Fine-tuning (e.g. `pi05_libero`) needs the data as a LeRobot dataset. Which conversion script
to use depends on which raw LIBERO data you have:

- **Raw benchmark HDF5 files** (`libero_10`, `libero_90`, `libero_spatial`, `libero_object`,
  `libero_goal`, e.g. from the original
  [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO) or a `libero_100.zip`
  download -- one `*_demo.hdf5` file per task, with `data/demo_N/obs/agentview_rgb` etc.
  inside): use `convert_libero_hdf5_to_lerobot.py`.

  ```bash
  uv run examples/libero/convert_libero_hdf5_to_lerobot.py \
      --data-dir /path/to/libero_90 \
      --repo-name your_hf_username/libero_90
  ```

- **RLDS/TFDS re-release** (`openvla/modified_libero_rlds` on Hugging Face, loaded via
  `tensorflow_datasets`): use `convert_libero_data_to_lerobot.py` (requires
  `uv pip install tensorflow tensorflow_datasets`).

Both scripts write to `$HF_LEROBOT_HOME` and produce the same `image` / `wrist_image` / `state`
(8-dim) / `actions` (7-dim) feature schema expected by `LeRobotLiberoDataConfig` in
`src/openpi/training/config.py` and `LiberoInputs`/`LiberoOutputs` in
`src/openpi/policies/libero_policy.py`. Point your `TrainConfig`'s `repo_id` at whatever
`--repo-name` you chose, then compute norm stats and train as described in the top-level
[README](../../README.md#fine-tuning-base-models-on-your-own-data).

## Splitting long-horizon demos into sub-skills

LIBERO-10 ("libero-long")'s tasks are each really 2-4 short sub-skills back to back (e.g. "pick
up the alphabet soup" / "place the alphabet soup in the basket" / "pick up the cream cheese box"
/ "place the cream cheese box in the basket"). Two scripts turn a directory of raw HDF5 files
into a LeRobot dataset of those sub-skills as separate, short-instruction episodes instead of one
long-instruction episode per demo:

1. **`annotate_subskill_boundaries.py`** -- for every demo, finds which frames belong to which
   sub-skill and writes `subskill_boundaries.json` next to the HDF5 files. Boundaries come from
   LIBERO's own task-success machinery, not a hand-rolled heuristic: "place"/"open"/"close"/"turn
   on" boundaries reuse the exact BDDL predicate classes LIBERO's `_check_success()` evaluates
   every step; "pick up" boundaries use gripper-object contact plus a real height rise (robosuite
   provides this style of check for its own tasks' reward shaping, e.g. `_check_grasp`, though
   this script uses the plainer contact+lift version -- see the script's docstring for why).
   Requires the LIBERO env (Python 3.8 + robosuite + mujoco, no GPU/rendering needed since it
   only replays recorded states for physics, not images):

   ```bash
   python examples/libero/annotate_subskill_boundaries.py \
       --data-dir /path/to/libero_10 \
       --out /path/to/libero_10/subskill_boundaries.json
   ```

   `--task-set` selects which sub-skill taxonomy/source-file mapping to use (default
   `libero_10`, the tasks above). `--task-set libero_90_lilo22` instead annotates
   [LiLo-VLA](https://yy-gx.github.io/LiLo-VLA/static/pdfs/appendix.pdf)'s 22-skill atomic
   library (its Table III), mapped onto 13 real LIBERO-90 HDF5 files rather than LIBERO-10's --
   LIBERO-90's tasks are already short/atomic (one pick + one place, or a single open/close/
   turn-on), so none of them need LIBERO-10's "transport" padding skill; a file just gets as many
   sub-skills as it genuinely has (1 or 2):

   ```bash
   python examples/libero/annotate_subskill_boundaries.py \
       --task-set libero_90_lilo22 \
       --data-dir /path/to/libero_90 \
       --out /path/to/libero_90/subskill_boundaries.json
   ```

   Every boundary is resolved to a concrete frame range even when detection genuinely can't find
   a signal (e.g. a demo's recording ends before any visible sign of release) -- those get a
   `detection_failed: true` flag rather than a silent wrong guess, so treat that sub-skill's cut
   point as an approximation, not a precise one.

   You can visualize the result with [hdf5-lerobot-visualizer](../../../hdf5-lerobot-visualizer)
   (a separate Qt GUI project): opening a directory that has a `subskill_boundaries.json` next to
   its HDF5 files shows a colored timeline of the 4 sub-skills alongside the video, a label for
   whichever sub-skill the current frame belongs to, and matching shaded regions on the
   trajectory-signal plot.

2. **`convert_libero_subskills_to_lerobot.py`** -- does the actual cutting, using the JSON from
   step 1. Each sub-skill segment becomes its own LeRobot episode, labeled with that sub-skill's
   short prompt instead of the demo's full instruction. Runs in the openpi (JAX/LeRobot) env, not
   the LIBERO one:

   ```bash
   uv run examples/libero/convert_libero_subskills_to_lerobot.py \
       --data-dir /path/to/libero_10 \
       --repo-name your_hf_username/libero_10_subskills
   ```

   Since a sub-skill segment's end is usually an artificial mid-demo cut (not the original
   episode's natural end), its last few real frames would otherwise come up short of a full
   `action_horizon`-length training target. The script pads every segment's end with `--pad-steps`
   (default 10, matching `pi05_libero`'s `action_horizon`) no-op frames: repeated last
   image/state (nothing moves) and a "stay put" action, which differs by action representation
   (`--action-type`, default `delta` -- matches this dataset, confirmed against the raw actions
   array in `convert_libero_hdf5_to_lerobot.py`'s docstring): zero pose-delta with the gripper
   channel held at its last commanded value for `delta`, vs. the entire last action repeated
   verbatim for `absolute`. See the script's docstring for the full reasoning.

## Visualizing prompt attention

`visualize_attention.py` runs one LIBERO episode and renders five panels side-by-side, written out
as one video: **rollout | base-camera attention overlay | wrist-camera attention overlay | word
value heatmap | word rank heatmap**.

**Language panels** (rendered once per inferred action chunk) -- **rows are every transformer
layer**, columns are the prompt's words:

- **Value heatmap**: each word's relative share of attention (renormalized per row, since the
  raw values are a tiny slice of a 800+-position softmax and would otherwise render as solid
  color); the title also shows what % of *total* attention (images included) landed on language
  at all.
- **Rank heatmap**: same data, but showing each word's rank (1 = most attended) instead of its
  value. Ranks always span the full 1..N range regardless of how close together the underlying
  values are, so this stays readable even when the value heatmap doesn't -- at the cost of
  showing only *ordering*, not magnitude (a landslide and a near-tie for 1st look identical here).

Every layer is shown at once (heads are still averaged away server-side, but never layers --
different layers attend to very different things, so blending them together would wash out
whichever pattern is actually informative; this is exactly why you may see only one or two rows
light up while the rest look dim -- that's often real, not a bug: only a few layers may specialize
in language grounding at all). The row axis that's collapsed instead is the action-chunk's
predicted steps (averaged over): within one chunk, every step shares the same prompt and
observation, so which *layer* attends to which word is the far more informative axis to show.

**Image panels**: the base (agentview) and wrist camera frames, each with the model's attention
over that camera's visual patch tokens alpha-blended on top as a Grad-CAM-style heatmap
(`--args.image-attn-alpha`, default 0.45, controls blend strength). Unlike the language panels,
this *does* average over layers -- there's no natural axis to lay separate per-layer overlays out
along, and the goal here is qualitative: watching across the rollout whether the hotspot tracks a
moved/manipulated object, or (as found for language) stays fixed regardless of what's actually in
view. The overlay is recomputed once per inferred action chunk (same as the word heatmaps) and
held fixed across the `--args.replan-steps` env steps it governs, but re-rendered onto that step's
*current* camera frame each time, so the underlying scene keeps moving under a chunk-stale
heatmap -- this is expected, not a bug.

Like the standard eval above, this is a two-terminal, client-server setup -- LIBERO's environment
and the full openpi/JAX model package have
hard-conflicting pinned dependencies (LIBERO needs Python 3.8 + `torch==1.11.0+cu113`; openpi
needs Python >=3.11 + `torch==2.7.1`) and cannot be installed together in one process, which is
why the model is always served separately. The only difference from the standard eval is which
server script and which client script you run:

### Without Docker

**Terminal 1** (server -- note `serve_policy_with_attention.py`, not `serve_policy.py`):
```bash
uv run scripts/serve_policy_with_attention.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```
(flags like `--num-denoising-steps` or `--port` must come *before* the `policy:checkpoint`
subcommand, not after -- tyro scopes trailing flags to the subcommand itself)

**Terminal 2** (client, in the LIBERO env):
```bash
python examples/libero/visualize_attention.py --args.task-suite-name libero_10 --args.task-id 0
```

### With Docker

Same `compose.yml` as the standard eval, just pointing each container at the attention-viz
script instead of the default via `SERVER_SCRIPT`/`CLIENT_SCRIPT` (both default to the standard
`serve_policy.py`/`main.py` when unset, so this doesn't affect normal usage):

```bash
sudo xhost +local:docker

SERVER_SCRIPT=scripts/serve_policy_with_attention.py \
SERVER_ARGS="policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero" \
CLIENT_SCRIPT=examples/libero/visualize_attention.py \
CLIENT_ARGS="--args.task-suite-name libero_10 --args.task-id 0" \
    docker compose -f examples/libero/compose.yml up --build
```

The `runtime` container's working directory (`/app`) is the repo root itself (via the
`$PWD:/app` mount), so the client's default `--video-out-path data/libero/attention_videos` is
already a path inside your repo checkout on the host -- no extra mount or override needed.

This uses `Pi0.sample_actions_with_attention` (`src/openpi/models/pi0.py`) via a dedicated
`AttentionCapturingPolicy` (`src/openpi/policies/attention_policy.py`) -- both return
actions bit-for-bit identical to the standard `sample_actions`/`Policy` path, just with the
attention weights attached, so none of this affects training or normal inference. The extra
`attn`/`prefix_len`/`has_state_token` fields ride over the same websocket protocol as `actions`
(no wire-format changes needed); the client re-tokenizes the prompt itself (downloading the same
public tokenizer file directly over HTTPS) to recover word boundaries, so it needs no `openpi`
dependency at all -- just `sentencepiece` and `matplotlib` (add `sentencepiece` to
`examples/libero/requirements.in` and recompile `requirements.txt` if it's not already installed
in your LIBERO venv).

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
