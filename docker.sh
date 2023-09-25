#!/bin/bash

if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
  CUDA_VISIBLE_DEVICES=all
fi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $BASE_DIR

DOCKER_IMAGE="$HOSTNAME/nerf"
echo $DOCKER_IMAGE

if [ $# -eq 0 ]; then
  SET_USER_ID=""
else
  SET_USER_ID="-u $(id -u):$(id -g)"
fi

docker run -it --rm                                   \
  $SET_USER_ID                                        \
  --gpus $CUDA_VISIBLE_DEVICES --privileged=true      \
  --device /dev/nvidia0                               \
  --device /dev/nvidia-uvm                            \
  --device /dev/nvidia-uvm-tools                      \
  --device /dev/nvidiactl                             \
  --device /dev/nvidia-modeset                        \
  -v "$BASE_DIR:/project"                             \
  -p 8080:8080                                        \
  -p 8081:8081                                        \
  -p 6006:6006                                        \
  -e PYTHONUNBUFFERED=1                               \
  --ipc=host                                          \
  $DOCKER_IMAGE $*
