"""Webcam access with automatic cleanup."""
import cv2 as cv


class WebcamManager:
    """Context manager for webcam operations with automatic resource management."""

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_opened = False

    def __enter__(self):
        """Open the webcam when entering the context."""
        print("Initializing webcam...")
        self.cap = cv.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        # Set camera properties for better performance
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv.CAP_PROP_FPS, self.fps)

        self.is_opened = True
        print(f"Webcam initialized: {self.width}x{self.height} @ {self.fps}fps")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the webcam when leaving the context."""
        print("Cleaning up webcam resources...")
        if self.cap and self.is_opened:
            self.cap.release()
        cv.destroyAllWindows()
        self.is_opened = False
        print("Webcam cleanup completed.")

        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions

    def read_frame(self):
        """Read a BGR frame from the webcam."""
        if not self.is_opened or not self.cap:
            raise RuntimeError("Webcam not initialized. Use within 'with' statement.")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam.")

        return frame

    def is_camera_opened(self):
        """Check if camera is properly opened."""
        return self.is_opened and self.cap and self.cap.isOpened()
