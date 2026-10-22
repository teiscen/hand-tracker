"""Webcam loop and the `hand-tracker` command."""
import argparse

import cv2 as cv

from .camera import WebcamManager
from .capture import FrameRecorder, generate_labelled_data
from .landmarker import HandLandmarkerManagerLive
from .overlay import apply_transforms


def process_frame(webcam, hand_manager, recorder):
    """Show one webcam frame with landmarks and handle key presses.

    Returns False when the loop should stop.
    """
    try:
        bgr_frame = webcam.read_frame()
    except RuntimeError as e:
        print(f"Error: {e}")
        return False

    cv.imshow('Hand Landmark Detection', apply_transforms(bgr_frame, hand_manager))

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        return False
    if key == ord('r'):
        recorder.start()

    frames = recorder.feed(bgr_frame)
    if frames is not None:
        generate_labelled_data(frames)

    return True


def main(argv=None):
    """Entry point for the `hand-tracker` command."""
    parser = argparse.ArgumentParser(
        prog="hand-tracker",
        description="Live hand tracking from a webcam. "
                    "Press 'q' to quit, 'r' to capture 4 frames and write output.txt.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="capture height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="capture fps (default: 30)")
    args = parser.parse_args(argv)

    try:
        with WebcamManager(camera_id=args.camera, width=args.width,
                           height=args.height, fps=args.fps) as webcam:
            with HandLandmarkerManagerLive() as hand_manager:
                print("Starting webcam feed. Press 'q' to quit.")
                recorder = FrameRecorder()

                while webcam.is_camera_opened():
                    if not process_frame(webcam, hand_manager, recorder):
                        break

    except RuntimeError as e:
        print(f"Webcam error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")

    print("Application stopped.")


if __name__ == "__main__":
    main()
