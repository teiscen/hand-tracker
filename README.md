# Hand Tracker

Real-time hand tracking from a webcam using MediaPipe and OpenCV. It extracts finger joint angles, extension, spread and thumb opposition, the inputs for driving a robotic hand. Project for SFU CMPT 310.

## Install

Needs a webcam and Python 3.9 to 3.12 (tested on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Run

```bash
hand-tracker                     # or: python -m hand_tracker
hand-tracker --camera 1          # pick a different camera
hand-tracker --help
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Capture 4 frames, smooth them, and write the features to `output.txt` in the current folder |

Only the right hand is used for features.

## How it works

```
Camera -> OpenCV (mirror, overlay) -> MediaPipe Hand Landmarker (21 points per hand)
       -> features (joint angles, extension, spread, thumb opposition)
       -> Savitzky-Golay smoothing -> [planned: model] -> ROS 2 / Gazebo
```

The pipeline diagram is in `Extraneous/Pipeline.png`.

## Project layout

| Path | What it is |
|------|------------|
| `hand_tracker/app.py` | Webcam loop and the `hand-tracker` command |
| `hand_tracker/camera.py` | Webcam wrapper |
| `hand_tracker/landmarker.py` | MediaPipe wrapper (live stream or single image) |
| `hand_tracker/overlay.py` | Draws the landmarks on the video |
| `hand_tracker/features.py` | Landmarks to joint angles, extension, spread and thumb opposition |
| `hand_tracker/smoothing.py` | Savitzky-Golay smoothing over a short clip |
| `hand_tracker/capture.py` | Records 4 frames and writes the `output.txt` report |
| `hand_tracker/report.py` | Text formatting for the report |
| `hand_tracker/models/` | The MediaPipe `hand_landmarker.task` model |
| `hand_tracker/ros/` | Optional ROS 2 publisher and the robot hand URDF |
| `hand_tracker/ml/` | Optional autoencoder experiment |
| `Extraneous/` | Reports, diagrams and notes |

## Optional parts

**ROS 2.** `hand_tracker/ros/robot_controller.py` needs `rclpy` from your ROS 2 install (it is not on pip). Source ROS 2, then run `python -m hand_tracker.ros.robot_controller`. For now it publishes fixed example joint angles, not live tracking data.

**Machine learning.** `pip install -e ".[ml]"` adds PyTorch. `hand_tracker/ml/auto_encoder.py` is an unfinished experiment and does not run yet.

## Status

Cleanup in progress. Planned next: tests for the feature math, a separate window for live data and test results, and replacing `output.txt` with saved recordings.
