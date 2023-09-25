#!/bin/bash

if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
  CUDA_VISIBLE_DEVICES=all
fi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $BASE_DIR

DOCKER_IMAGE="$HOSTNAME/nerf"
if [[ -f "$BASE_DIR/Dockerfile" ]]; then
  if [[ "$(docker images -q "$DOCKER_IMAGE:latest" 2> /dev/null)" != "" ]]; then
    echo "renaming old image... $DOCKER_IMAGE:latest -> $DOCKER_IMAGE:old"
    docker tag "$DOCKER_IMAGE:latest" "$DOCKER_IMAGE:old"
  fi
  echo "building docker image... $DOCKER_IMAGE:latest"
  cd "$( dirname "${BASH_SOURCE[0]}" )"
  docker build --tag $DOCKER_IMAGE .
  if [[ "$(docker images -q "$DOCKER_IMAGE:old" 2> /dev/null)" != "" ]]; then
    echo "removing old image... $DOCKER_IMAGE:old"
    docker rmi "$DOCKER_IMAGE:old"
  fi
fi
