#!/bin/bash
set -e

cd "$(dirname "$0")"

# Allow Docker to talk to the X server
xhost +local:docker >/dev/null

# Build (first run only) and start
docker compose up -d --build

# Wait for the container to be ready
sleep 2

# Launch everything inside the container
docker compose exec hand-tracker bash -c '
  source /opt/ros/jazzy/setup.bash
  source /opt/venv/bin/activate
  cd /workspace/hand-tracker/Code
  ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(xacro '"'"'/workspace/hand-tracker/Code/Hand Urdf/BionicHand.xacro'"'"')" &
  RSP_PID=$!

  python3 robot_controller.py &
  CTRL_PID=$!

  sleep 1
  rviz2 -d /workspace/hand-tracker/Code/hand.rviz &
  RViz_PID=$!

  # Foreground: camera window. Press q to quit.
  python3 opencv.py || true

  kill $RSP_PID $CTRL_PID $RViz_PID 2>/dev/null || true
'
