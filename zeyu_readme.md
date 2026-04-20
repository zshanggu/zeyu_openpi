docker build -t openpi_server -f scripts/docker/serve_policy.Dockerfile .

sudo docker run -it \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --shm-size 8g \
  --network host \
  --gpus all \
  --ipc=host \
  -v /media:/media \
  -v /data:/data \
  -v /home:/home \
  -v ~/.cache/openpi:/openpi_assets \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi:/app \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  --user root \
  --name zeyu_openpi_aloha_sim \
  -e SERVER_ARGS="--env ALOHA_SIM" \
  openpi_server

sudo docker start -ai zeyu_openpi_aloha_sim

docker build -t aloha_sim -f examples/aloha_sim/Dockerfile .

docker run -it \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --network host \
  --gpus all \
  --ipc=host \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi:/app \
  --name zeyu_aloha_sim \
  aloha_sim

docker start -ai zeyu_aloha_sim

### Libero

git submodule update --init --recursive

docker build -t libero -f examples/libero/Dockerfile .

sudo docker run -it \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --shm-size 8g \
  --network host \
  --gpus all \
  --ipc=host \
  -v /media:/media \
  -v /data:/data \
  -v /home:/home \
  -v ~/.cache/openpi:/openpi_assets \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi:/app \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  --user root \
  --name zeyu_openpi_libero \
  -e SERVER_ARGS="--env LIBERO" \
  openpi_server

docker start -ai zeyu_openpi_libero

# pi0

sudo docker run -it \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --shm-size 8g \
  --network host \
  --gpus all \
  --ipc=host \
  -v /media:/media \
  -v /data:/data \
  -v /home:/home \
  -v ~/.cache/openpi:/openpi_assets \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi:/app \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  --user root \
  --name zeyu_openpi_libero_pi0 \
  -e SERVER_ARGS="policy:checkpoint --policy.config pi0_libero --policy.dir gs://openpi-assets/checkpoints/pi0_libero" \
  openpi_server

docker start -ai zeyu_openpi_libero_pi0

# LIBERO
docker run -it \
  --network host \
  --gpus all \
  --privileged \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi:/app \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi/data:/data \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY=$DISPLAY \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  --name zeyu_libero_sim \
  libero \
  bash -c "source /.venv/bin/activate && python examples/libero/main.py"

docker start -ai zeyu_libero_sim

# LIBERO-10 (long horizon)
docker run -it \
  --network host \
  --gpus all \
  --privileged \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi:/app \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi/data:/data \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY=$DISPLAY \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  --name zeyu_libero10_sim \
  libero \
  bash -c "source /.venv/bin/activate && python examples/libero/main.py --args.task-suite-name libero_10"

docker start -ai zeyu_libero10_sim

### DROID

sudo docker run -it \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --shm-size 8g \
  --network host \
  --gpus all \
  --ipc=host \
  -v /media:/media \
  -v /data:/data \
  -v /home:/home \
  -v ~/.cache/openpi:/openpi_assets \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi:/app \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  --user root \
  --name zeyu_openpi_droid \
  -e SERVER_ARGS="--env DROID" \
  openpi_server

### Non-finetuned
sudo docker run -it \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --shm-size 8g \
  --network host \
  --gpus all \
  --ipc=host \
  -v /media:/media \
  -v /data:/data \
  -v /home:/home \
  -v ~/.cache/openpi:/openpi_assets \
  -v ~/PHD_LAB/Course/CSCI662/zeyu_openpi:/app \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  --user root \
  --name zeyu_openpi_base \
  -e SERVER_ARGS="policy:checkpoint --policy.config=pi05_base --policy.dir=gs://openpi-assets/checkpoints/pi05_base" \
  openpi_server


# Train
1. Set up the environment

Installed uv via the official installer
Logged into HuggingFace (huggingface-cli login)
Downloaded the physical-intelligence/libero dataset (~1699 files)
2. Created a LoRA config

Added pi05_libero_lora to _CONFIGS in src/openpi/training/config.py
Uses paligemma_variant="gemma_2b_lora" + action_expert_variant="gemma_300m_lora" with ema_decay=None
Points pytorch_weight_path to the local converted checkpoint
3. Computed norm stats


uv run scripts/compute_norm_stats.py --config-name=pi05_libero_lora
4. Converted the JAX base checkpoint to PyTorch


uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/.../pi05_base \
    --config_name pi05_libero_lora \
    --output_path ~/.cache/openpi/.../pi05_base_pytorch
(Required copying transformers_replace/ files first)

5. Fixed the PyTorch trainer to freeze the backbone

Modified scripts/train_pytorch.py to detect LoRA configs and freeze paligemma_with_expert.paligemma, reducing optimizer state memory from ~15GB to ~2.4GB
6. Launched training


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run scripts/train_pytorch.py pi05_libero_lora \
    --exp_name my_pi05_libero_lora --overwrite --batch_size 4