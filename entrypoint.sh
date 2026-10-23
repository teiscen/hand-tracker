#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash

# Activate the venv, but clear PYTHONPATH to avoid ROS's Python path overriding
source /opt/venv/bin/activate
unset PYTHONPATH

exec "$@"
