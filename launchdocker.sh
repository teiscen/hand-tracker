#!/bin/bash

docker run -it --rm \
    --net=host \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PWD:/workspace/hand-tracker:ro" \
    osrf/ros:foxy-desktop
