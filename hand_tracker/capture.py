"""Record a short burst of frames and turn it into smoothed feature data."""
from contextlib import redirect_stdout

import cv2 as cv

from .features import create_example_data
from .landmarker import HandLandmarkerManagerImage
from .report import print_example_data, print_labelled_data
from .smoothing import create_labelled_example


class FrameRecorder:
    """Collects n consecutive frames after start() and hands them back when full."""

    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = []
        self.active = False

    def start(self):
        """Begin recording. Ignored if a recording is already in progress."""
        if not self.active:
            self.frames = []
            self.active = True

    def feed(self, bgr_frame):
        """Add a frame while recording. Returns the list of frames once it is full, else None."""
        if not self.active:
            return None

        self.frames.append(bgr_frame.copy())  # copy so later frame reuse can't change it
        if len(self.frames) < self.n_frames:
            return None

        frames, self.frames, self.active = self.frames, [], False
        return frames


def generate_labelled_data(bgr_frame_list, report_path="output.txt"):
    """Detect hands in each frame, extract right-hand features, smooth them over the clip.

    A text report goes to report_path; the smoothed data is also returned
    (None if no right hand was found in any frame).
    """
    detections = []
    example_list = []
    labelled_data = None

    with open(report_path, 'w') as report:
        with redirect_stdout(report):
            try:
                with HandLandmarkerManagerImage() as image_hand_manager:
                    print(f"Processing {len(bgr_frame_list)} frames...")

                    for i, bgr_frame in enumerate(bgr_frame_list):
                        print(f"Processing frame {i+1}/{len(bgr_frame_list)}")
                        try:
                            rgb_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
                            detections.append(image_hand_manager.detect(rgb_frame))
                        except Exception as e:
                            print(f"Error processing frame {i+1}: {e}")
                            detections.append(None)

                    for i, detection_result in enumerate(detections):
                        print(f"Extracting features from frame {i+1}")
                        if detection_result is None:
                            print(f"Frame {i+1} processing failed")
                            continue
                        example = create_example_data(detection_result)
                        if example is None:
                            print(f"No right hand detected in frame {i+1}")
                            continue
                        print_example_data(example)
                        example_list.append(example)

                    if example_list:
                        print(f"\nCreating labelled data from {len(example_list)} valid frames...")
                        labelled_data = create_labelled_example(example_list)
                        print_labelled_data(labelled_data)
                    else:
                        print("No valid hand data found in any frames")

            except Exception as e:
                print(f"Error in generate_labelled_data: {e}")

    print(f"Labelled data generation complete. Check {report_path} for details.")
    return labelled_data
