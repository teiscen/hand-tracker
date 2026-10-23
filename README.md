# Bionic Hand Tracker

Live hand tracking → URDF model in RViz, all inside Docker.

## Requirements

- **Linux** host with X11 (GUI). macOS/Windows are not supported out of the box.
- Docker Engine + Docker Compose v2 (`docker compose` command).
- A webcam.

## Quick start

    ./start.sh

First run builds the image (~3–5 minutes). Later runs are instant.

A camera window opens and RViz shows the hand model. Move your hand — the
model should follow. Press `q` in the camera window to quit.

## If your webcam isn't /dev/video0

Find the correct device on the host:

    ls /dev/video*

Then run with:

    VIDEO_DEVICE=/dev/video1 ./start.sh

## If no windows appear

X11 permissions may not have been granted. Run once per login:

    xhost +local:docker

(already done by `start.sh`, but if it fails, run it manually)

## What runs inside

- `robot_state_publisher` — reads `BionicHand.xacro`, publishes `/robot_description` and TF
- `robot_controller.py` — subscribes to `/hand_landmarks`, computes joint angles, publishes `/joint_states`
- `opencv.py` — webcam + MediaPipe, publishes `/hand_landmarks`
- `rviz2` — visualization

## Stopping

    docker compose down

## Rebuilding after code changes

    docker compose build && ./start.sh
