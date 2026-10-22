"""MediaPipe hand landmarker wrapper: live-stream mode or single-image mode."""
import time
from pathlib import Path

import mediapipe as mp

# The model ships inside the package, so it is found no matter where you run from.
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

LIVE = "live"
IMAGE = "image"


class HandLandmarkerManager:
    """Context manager around a MediaPipe HandLandmarker.

    mode="live":  frames go in with detect_async(); read the newest result with
                  get_latest_result().
    mode="image": call detect(); the result is returned directly (batch processing).
    """

    def __init__(self,
                 mode=LIVE,
                 model_path=DEFAULT_MODEL_PATH,
                 num_hands=2,
                 min_detection_confidence=0.5,
                 min_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
        if mode not in (LIVE, IMAGE):
            raise ValueError(f"mode must be '{LIVE}' or '{IMAGE}', got {mode!r}")
        self.mode = mode
        self.model_path = str(model_path)
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Set in __enter__
        self.landmarker = None
        self.latest_result = None
        self.is_initialized = False

    def __enter__(self):
        """Create the MediaPipe landmarker when entering the context."""
        print(f"Initializing MediaPipe hand landmarker ({self.mode} mode)...")

        try:
            common = dict(
                base_options=BaseOptions(model_asset_path=self.model_path),
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            if self.mode == LIVE:
                options = HandLandmarkerOptions(
                    running_mode=VisionRunningMode.LIVE_STREAM,
                    result_callback=self._result_callback,
                    **common)
            else:
                options = HandLandmarkerOptions(
                    running_mode=VisionRunningMode.IMAGE, **common)

            self.landmarker = HandLandmarker.create_from_options(options)
            self.is_initialized = True

            print("Hand landmarker initialized successfully:")
            print(f"  - Model: {self.model_path}")
            print(f"  - Max hands: {self.num_hands}")
            print(f"  - Detection confidence: {self.min_detection_confidence}")

            return self

        except Exception as e:
            print(f"Failed to initialize hand landmarker: {e}")
            raise RuntimeError(f"Hand landmarker initialization failed: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release MediaPipe resources when leaving the context."""
        print("Cleaning up hand landmarker resources...")

        if self.landmarker and self.is_initialized:
            try:
                self.landmarker.close()
                print("Hand landmarker closed successfully.")
            except Exception as e:
                print(f"Error closing hand landmarker: {e}")

        self.landmarker = None
        self.latest_result = None
        self.is_initialized = False

        if exc_type:
            print(f"Exception occurred in hand landmarker: {exc_type.__name__}: {exc_val}")

        return False  # Don't suppress exceptions

    def _result_callback(self, result, image, timestamp_ms: int):
        """Called by MediaPipe (live mode) whenever a detection finishes."""
        self.latest_result = result

    def _require_ready(self, mode):
        if not self.is_initialized or not self.landmarker:
            raise RuntimeError("Hand landmarker not initialized. Use within 'with' statement.")
        if self.mode != mode:
            method = "detect_async()" if mode == LIVE else "detect()"
            raise RuntimeError(f"{method} needs a landmarker created with mode='{mode}'.")

    def detect_async(self, rgb_frame):
        """Live mode: submit an RGB frame; the result arrives via get_latest_result()."""
        self._require_ready(LIVE)

        try:
            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
            timestamp = int(time.time() * 1000)
            self.landmarker.detect_async(mp_image, timestamp)

        except Exception as e:
            print(f"Error during hand detection: {e}")
            raise

    def detect(self, rgb_frame):
        """Image mode: detect hands in an RGB frame and return the result."""
        self._require_ready(IMAGE)

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self.latest_result = self.landmarker.detect(mp_image)
            return self.latest_result

        except Exception as e:
            print(f"Error during IMAGE mode hand detection: {e}")
            raise

    def get_latest_result(self):
        """Get the most recent detection result."""
        return self.latest_result

    def has_hands(self):
        """Check if hands were detected in the latest result."""
        return (self.latest_result is not None and
                self.latest_result.hand_landmarks is not None and
                len(self.latest_result.hand_landmarks) > 0)

    def get_hand_count(self):
        """Get the number of detected hands."""
        if not self.has_hands():
            return 0
        return len(self.latest_result.hand_landmarks)

    def is_ready(self):
        """Check if the landmarker is ready for detection."""
        return self.is_initialized and self.landmarker is not None


class HandLandmarkerManagerLive(HandLandmarkerManager):
    """Live-stream landmarker (kept so existing call sites don't change)."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, **kwargs):
        super().__init__(mode=LIVE, model_path=model_path, **kwargs)


class HandLandmarkerManagerImage(HandLandmarkerManager):
    """Single-image landmarker (kept so existing call sites don't change)."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, **kwargs):
        super().__init__(mode=IMAGE, model_path=model_path, **kwargs)
