"""Turn MediaPipe hand landmarks into the features used to drive a robotic hand."""
import math

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Landmark index triplets (a, joint, b): the angle is measured at the middle landmark.
# Each finger has two measured joints.
JOINT_TRIPLETS = {
    "thumb":  ((1, 2, 3),    (2, 3, 4)),
    "index":  ((5, 6, 7),    (6, 7, 8)),
    "middle": ((9, 10, 11),  (10, 11, 12)),
    "ring":   ((13, 14, 15), (14, 15, 16)),
    "pinky":  ((17, 18, 19), (18, 19, 20)),
}

FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

# Adjacent finger pairs whose spread angle is measured.
SPREAD_PAIRS = (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))
SPREAD_NAMES = tuple(f"{a}_{b}" for a, b in SPREAD_PAIRS)


def vector_angle(v1, v2):
    """Angle in degrees between two vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


# Joint Angles
def calculate_joint_angle(p1, p2, p3):
    """Angle at joint p2 between points p1-p2-p3."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    return vector_angle(v1, v2)


def calculate_joint_angles(hand_landmarks):
    """Two joint angles per finger, e.g. {'index_angles': [mcp_pip_dip, pip_dip_tip]}."""
    return {
        f"{finger}_angles": [
            calculate_joint_angle(*(hand_landmarks[i] for i in triplet))
            for triplet in triplets
        ]
        for finger, triplets in JOINT_TRIPLETS.items()
    }


# Finger Extension
def calculate_finger_extension(hand_landmarks):
    """Distance from the wrist to each fingertip (in image-normalized x/y)."""
    wrist = hand_landmarks[0]
    extensions = {}
    for finger, tip_idx in FINGERTIPS.items():
        tip = hand_landmarks[tip_idx]
        extensions[finger] = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
    return extensions


# Finger Spread
def calculate_finger_spread(hand_landmarks):
    """Angle between the wrist-to-fingertip directions of adjacent fingers."""
    wrist = hand_landmarks[0]
    vectors = {}
    for finger in ("index", "middle", "ring", "pinky"):
        tip = hand_landmarks[FINGERTIPS[finger]]
        vectors[finger] = np.array([tip.x - wrist.x, tip.y - wrist.y])
    return {f"{a}_{b}": vector_angle(vectors[a], vectors[b]) for a, b in SPREAD_PAIRS}


# Thumb Opposition
def calculate_thumb_opposition(hand_landmarks):
    """Distance between the thumb tip and the index fingertip."""
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    return math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)


def create_example_data(detection_result):
    """Extract features from a detection result for the right hand only."""
    if not detection_result or not detection_result.hand_landmarks:
        return None

    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
        try:
            if detection_result.handedness[i][0].category_name == "Right":
                return {
                    'joint_angles':      calculate_joint_angles(hand_landmarks),
                    'finger_extensions': calculate_finger_extension(hand_landmarks),
                    'finger_spread':     calculate_finger_spread(hand_landmarks),
                    'thumb_opposition':  calculate_thumb_opposition(hand_landmarks),
                }
        except Exception as e:
            print(f"Error processing hand {i}: {e}")
            continue

    return None  # No right hand found
