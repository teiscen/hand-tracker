"""Draw detected hand landmarks over webcam frames."""
import cv2 as cv
import mediapipe as mp

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def flip_horizontal(rgb_frame):
    """Mirror the frame so it behaves like a mirror."""
    return cv.flip(rgb_frame, 1)


def draw_landmarks(rgb_frame, detection_result):
    """Draw hand connections and numbered landmarks onto the frame (in place)."""
    if not detection_result or not detection_result.hand_landmarks:
        return rgb_frame

    height, width, _ = rgb_frame.shape
    for hand_landmarks in detection_result.hand_landmarks:
        # Convert normalized coordinates to pixel coordinates
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

        for start, end in HAND_CONNECTIONS:
            cv.line(rgb_frame, points[start], points[end], (0, 255, 0), 2)

        for idx, (x, y) in enumerate(points):
            cv.circle(rgb_frame, (x, y), 5, (0, 0, 255), -1)
            cv.putText(rgb_frame, str(idx), (x + 5, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return rgb_frame


def apply_transforms(bgr_frame, hand_manager):
    """Mirror a webcam frame, send it for detection, and draw the latest landmarks on it."""
    rgb_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
    rgb_frame = flip_horizontal(rgb_frame)

    hand_manager.detect_async(rgb_frame)
    rgb_frame = draw_landmarks(rgb_frame, hand_manager.get_latest_result())

    return cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
