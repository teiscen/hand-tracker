"""Plain-text reports of per-frame and smoothed features."""
from .features import FINGERS, SPREAD_NAMES


def print_example_data(example_features, file=None):
    """Print one frame's features."""
    if example_features is None:
        print("No features to display (example_features is None)", file=file)
        return

    # Joint angles (most important for servos)
    print("\nJOINT ANGLES (for servo control):", file=file)
    for finger, angles in example_features['joint_angles'].items():
        name = finger.replace('_angles', '').upper()
        print(f"  {name:<8}: {[f'{float(a):.1f}' for a in angles]}", file=file)

    # Finger extensions (for grip strength)
    print("\nFINGER EXTENSIONS:", file=file)
    for finger, extension in example_features['finger_extensions'].items():
        print(f"  {finger.capitalize():<8}: {float(extension):.3f}", file=file)

    # Finger spread (for object grasping)
    print("\nFINGER SPREAD:", file=file)
    for spread, angle in example_features['finger_spread'].items():
        print(f"  {spread:<15}: {float(angle):.1f}", file=file)

    # Thumb opposition (for precision grip)
    print(f"\nTHUMB OPPOSITION: {float(example_features['thumb_opposition']):.3f}", file=file)


def print_labelled_data(labelled_features):
    """Print smoothed features: one list of values per measurement, one value per frame."""
    if labelled_features is None:
        print("No labelled features to display")
        return

    print("\nLABELLED JOINT ANGLES (smoothed):")
    for finger in FINGERS:
        series_per_joint = labelled_features['joint_angles'][f"{finger}_angles"]
        joints = [str([f"{float(a):.1f}" for a in series]) for series in series_per_joint]
        print(f"  {finger.upper():<8}: {joints}")

    print("\nLABELLED FINGER EXTENSIONS (smoothed):")
    for finger in FINGERS:
        series = labelled_features['finger_extensions'][f"{finger}_extensions"]
        print(f"  {finger.capitalize():<8}: {[f'{float(e):.3f}' for e in series]}")

    print("\nLABELLED FINGER SPREAD (smoothed):")
    for name in SPREAD_NAMES:
        series = labelled_features['finger_spread'][f"{name}_spread"]
        print(f"  {name + '_spread':<20}: {[f'{float(a):.1f}' for a in series]}")

    print("\nLABELLED THUMB OPPOSITION (smoothed):")
    series = labelled_features['thumb_opposition']
    print(f"  Thumb Opposition: {[f'{float(t):.3f}' for t in series]}")
