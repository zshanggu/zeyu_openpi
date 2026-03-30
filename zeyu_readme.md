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

docker start -ai zeyu_libero_sim

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