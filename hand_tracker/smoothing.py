"""Smooth per-frame features over a short clip (Savitzky-Golay filter)."""
import numpy as np
from scipy.signal import savgol_filter

from .features import FINGERS, SPREAD_NAMES


def _smooth(values, window_length, polyorder):
    """Savitzky-Golay smoothing. With window_length=None the values pass through unchanged."""
    if window_length is None:
        return np.asarray(values, dtype=float)
    return savgol_filter(values, window_length, polyorder)


def label_joint_angles(data_list, window_length, polyorder):
    labelled = {}
    for finger in FINGERS:
        key = f"{finger}_angles"
        # One time series per joint: [joint 1 over frames, joint 2 over frames]
        per_joint = zip(*(data['joint_angles'][key] for data in data_list))
        labelled[key] = [_smooth(list(series), window_length, polyorder) for series in per_joint]
    return labelled


def label_finger_extension(data_list, window_length, polyorder):
    return {
        f"{finger}_extensions": _smooth(
            [data['finger_extensions'][finger] for data in data_list], window_length, polyorder)
        for finger in FINGERS
    }


def label_finger_spread(data_list, window_length, polyorder):
    return {
        f"{name}_spread": _smooth(
            [data['finger_spread'][name] for data in data_list], window_length, polyorder)
        for name in SPREAD_NAMES
    }


def label_thumb_opposition(data_list, window_length, polyorder):
    return _smooth([data['thumb_opposition'] for data in data_list], window_length, polyorder)


def create_labelled_example(example_features_list):
    """Smooth a list of per-frame feature dicts into one dict of time series.

    The window is 5 frames (or 3 for 3-4 frames). With fewer than 3 frames there is
    nothing to smooth, so the raw values come back in the same shape.
    """
    data_length = len(example_features_list)

    # Window length must be odd and <= data length
    if data_length >= 5:
        window_length = 5
    elif data_length >= 3:
        window_length = 3
    else:
        print(f"Warning: Only {data_length} frames, skipping smoothing")
        window_length = None

    polyorder = None
    if window_length is not None:
        polyorder = min(2, window_length - 1)  # polyorder must be < window_length
        print(f"Using window_length={window_length}, polyorder={polyorder} for {data_length} frames")

    return {
        'joint_angles':      label_joint_angles(example_features_list, window_length, polyorder),
        'finger_extensions': label_finger_extension(example_features_list, window_length, polyorder),
        'finger_spread':     label_finger_spread(example_features_list, window_length, polyorder),
        'thumb_opposition':  label_thumb_opposition(example_features_list, window_length, polyorder),
    }
