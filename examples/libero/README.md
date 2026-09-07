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

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
