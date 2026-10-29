"""Turn one skeleton window (frames, 17, 3) into a small feature vector."""
import warnings
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12   # COCO keypoint indices
L_ANKLE, R_ANKLE = 15, 16
IMG_ASPECT = 4 / 3   # width / height of Le2i videos (320x240); x and y are normalized separately

FEATURE_NAMES = [
    "hip_drop",            # how far the hips moved down during the window (+ = down)
    "hip_max_speed",       # fastest downward hip movement between two frames
    "torso_angle_end",     # torso angle from vertical at the end (0 = upright, 90 = lying)
    "torso_angle_change",  # how much the torso tilted during the window
    "aspect_end",          # body width / height at the end (lying = wide)
    "legs_angle_end",      # hip->ankle angle from vertical at the end (0 = on feet, 90 = on floor)
]


def _mean(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices just give NaN
        return np.nanmean(x)


def _max(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(x) if np.any(~np.isnan(x)) else np.nan


def window_features(kp, k=5):
    """kp: (frames, 17, 3) with normalized x, y and confidence; NaN = missing."""
    shoulder = (kp[:, L_SH, :2] + kp[:, R_SH, :2]) / 2      # NaN if either shoulder missing
    hip = (kp[:, L_HIP, :2] + kp[:, R_HIP, :2]) / 2
    hip_y = hip[:, 1]                                       # y grows downward in images

    torso = hip - shoulder
    torso_dx = torso[:, 0] * IMG_ASPECT                     # undo the different x/y scaling
    angle = np.degrees(np.arctan2(np.abs(torso_dx), np.abs(torso[:, 1])))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        width = (np.nanmax(kp[:, :, 0], axis=1) - np.nanmin(kp[:, :, 0], axis=1)) * IMG_ASPECT
        height = np.nanmax(kp[:, :, 1], axis=1) - np.nanmin(kp[:, :, 1], axis=1)
    aspect = np.where(height > 0, width / height, np.nan)

        # Legs: line from mid-hip to the ankles. Bending keeps legs vertical, a fall doesn't.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ankle = np.nanmean(kp[:, [L_ANKLE, R_ANKLE], :2], axis=1)   # one visible ankle is enough
    legs = ankle - hip
    legs_angle = np.degrees(np.arctan2(np.abs(legs[:, 0] * IMG_ASPECT), np.abs(legs[:, 1])))

    return np.array([
        _mean(hip_y[-k:]) - _mean(hip_y[:k]),
        _max(np.diff(hip_y)),
        _mean(angle[-k:]),
        _mean(angle[-k:]) - _mean(angle[:k]),
        _mean(aspect[-k:]),
        _mean(legs_angle[-k:]),
    ], dtype=np.float32)