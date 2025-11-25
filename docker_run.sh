#!/bin/bash
# start_docker.sh - 放在 final_project/ 目录下

docker run -it --rm --gpus all \
  -v /root/Robotic_gzs/final_project/src:/workspace/src \
  -v /root/Robotic_gzs/final_project/isaacgym:/workspace/isaacgym \
  --ipc=host \
  --name isaacgym-container \
  isaacgym:latest bash