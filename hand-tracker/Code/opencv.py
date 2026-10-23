import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import mediapipe as mp
import cv2 as cv
from hand_landmarker import HandLandmarkerManagerLive


class LandmarkPublisher(Node):
    def __init__(self):
        super().__init__("landmark_publisher")
        self.pub = self.create_publisher(String, "/hand_landmarks", 10)

    def publish_result(self, detection_result):
        if not detection_result or not detection_result.hand_landmarks:
            return
        msg = String()
        msg.data = json.dumps(
            [
                [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]
                for hand in detection_result.hand_landmarks
            ]
        )
        self.pub.publish(msg)


class Transforms:
    def apply_transforms(bgr_frame, hand_manager):
        rgb_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
        rgb_frame = Transforms.draw_flipped(rgb_frame)
        rgb_frame = Transforms.draw_landmarks(rgb_frame, hand_manager)
        bgr_frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
        return bgr_frame

    def draw_landmarks(rgb_frame, hand_manager):
        hand_manager.detect_async(rgb_frame)
        detection_result = hand_manager.get_latest_result()

        if not detection_result or not detection_result.hand_landmarks:
            return rgb_frame

        for hand_landmarks in detection_result.hand_landmarks:
            height, width, _ = rgb_frame.shape
            landmark_points = []
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmark_points.append((x, y))

            HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
            for connection in HAND_CONNECTIONS:
                start_point = landmark_points[connection[0]]
                end_point = landmark_points[connection[1]]
                cv.line(rgb_frame, start_point, end_point, (0, 255, 0), 2)

            for idx, (x, y) in enumerate(landmark_points):
                cv.circle(rgb_frame, (x, y), 5, (0, 0, 255), -1)
                cv.putText(
                    rgb_frame,
                    str(idx),
                    (x + 5, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )

        return rgb_frame

    def draw_flipped(rgb_frame):
        return cv.flip(rgb_frame, 1)


class WebcamManager:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_opened = False

    def __enter__(self):
        print("Initializing webcam...")
        self.cap = cv.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv.CAP_PROP_FPS, self.fps)
        self.is_opened = True
        print(f"Webcam initialized: {self.width}x{self.height} @ {self.fps}fps")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Cleaning up webcam resources...")
        if self.cap and self.is_opened:
            self.cap.release()
        cv.destroyAllWindows()
        self.is_opened = False
        print("Webcam cleanup completed.")
        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False

    def read_frame(self):
        if not self.is_opened or not self.cap:
            raise RuntimeError("Webcam not initialized. Use within 'with' statement.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam.")
        return frame

    def is_camera_opened(self):
        return self.is_opened and self.cap and self.cap.isOpened()


def webcam_loop(webcam_manager, hand_manager, landmark_pub):
    try:
        bgr_frame = webcam_manager.read_frame()
    except RuntimeError as e:
        print(f"Error: {e}")
        return False

    updated_bgr_frame = Transforms.apply_transforms(bgr_frame, hand_manager)
    cv.imshow("Hand Landmark Detection", updated_bgr_frame)

    # Publish whatever MediaPipe currently has (live stream callback)
    landmark_pub.publish_result(hand_manager.get_latest_result())

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    return True


def main():
    rclpy.init()
    landmark_pub = LandmarkPublisher()

    try:
        with WebcamManager(camera_id=0, width=640, height=480, fps=30) as webcam:
            with HandLandmarkerManagerLive() as hand_manager:
                print("Starting webcam feed. Press 'q' to quit.")
                while webcam.is_camera_opened():
                    rclpy.spin_once(landmark_pub, timeout_sec=0)
                    if not webcam_loop(webcam, hand_manager, landmark_pub):
                        break
    except RuntimeError as e:
        print(f"Webcam error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        landmark_pub.destroy_node()
        rclpy.shutdown()
    print("Application stopped.")


if __name__ == "__main__":
    main()
