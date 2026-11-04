#!/bin/bash
# Runs on the Jetson itself (NOT in Docker).
# Reads the CSI ribbon camera and sends it as a video stream on port 5000.
# The Docker container (live_detect.py) reads that stream.

gst-launch-1.0 -e \
  nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM), width=640, height=480, framerate=25/1' ! \
  nvvidconv ! nvv4l2h264enc insert-sps-pps=true iframeinterval=25 idrinterval=25 ! \
  h264parse config-interval=-1 ! mpegtsmux ! \
  tcpserversink host=127.0.0.1 port=5000 sync=false
