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
  --name zeyu_openpi \
  -e SERVER_ARGS="--env ALOHA_SIM" \
  openpi_server

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