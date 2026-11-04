#!/bin/bash
# Starts the fall detector with a preview window on the Jetson's screen.
xhost +local:root
sudo docker run -it --rm --runtime nvidia --ipc=host --network host \
  --env-file falldet.env \
  -e HEADLESS=0 \
  -e OPENCV_FFMPEG_CAPTURE_OPTIONS="fflags;nobuffer|flags;low_delay" \
  -e TIMING=1 \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e QT_X11_NO_MITSHM=1 \
  -v /etc/localtime:/etc/localtime:ro \
  falldet
xhost -local:root