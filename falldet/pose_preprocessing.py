import math
import numpy as np


def interpolate_short_gaps(keypoints, fps, max_gap_seconds=0.12):
    """Fill short gaps within ONE window.

    Input:
        keypoints: array with shape (frames, 17, 3).
        The last dimension contains x, y, confidence.

    Returns:
        result: copy containing interpolated coordinates.
        interpolated: True where coordinates were estimated.
        remaining_missing: True where coordinates are still missing.

    Original input and confidence values remain unchanged.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite.")

    if not math.isfinite(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError("Gap limit must be finite and non-negative.")

    result = np.array(keypoints, dtype=np.float64, copy=True)

    if result.ndim != 3 or result.shape[1:] != (17, 3):
        raise ValueError("Expected shape: (frames, 17, 3).")

    max_gap_frames = math.floor(fps * max_gap_seconds + 1e-9)
    frame_count = len(result)

    missing = ~np.isfinite(result[:, :, :2]).all(axis=2)
    interpolated = np.zeros(missing.shape, dtype=bool)

    for joint in range(17):
        position = 0

        while position < frame_count:
            if not missing[position, joint]:
                position += 1
                continue

            start = position

            while position < frame_count and missing[position, joint]:
                position += 1

            end = position
            gap_length = end - start

            # Both endpoints must be inside this window.
            bounded = start > 0 and end < frame_count

            if not bounded or gap_length > max_gap_frames:
                continue

            before = result[start - 1, joint, :2].copy()
            after = result[end, joint, :2].copy()

            for frame in range(start, end):
                fraction = (frame - start + 1) / (gap_length + 1)

                result[frame, joint, :2] = (
                    (1 - fraction) * before + fraction * after
                )
                interpolated[frame, joint] = True

    remaining_missing = ~np.isfinite(
        result[:, :, :2]
    ).all(axis=2)

    return result, interpolated, remaining_missing


if __name__ == "__main__":
    # Small artificial example: three missing positions.
    example = np.zeros((5, 17, 3), dtype=float)

    example[0, 5, :2] = [0.1, 0.2]
    example[4, 5, :2] = [0.5, 0.6]
    example[1:4, 5, :2] = np.nan

    for fps in [25, 24]:
        result, filled, missing = interpolate_short_gaps(example, fps)

        print(f"\nFPS: {fps}")
        print("Joint observations interpolated:", int(filled.sum()))
        print("Joint observations still missing:", int(missing.sum()))
        print("Left shoulder coordinates:")
        print(result[:, 5, :2])