from pathlib import Path
import math
import numpy as np

from falldet.config import OUTPUTS_DIR
POSES_DIR = OUTPUTS_DIR / "poses"

WINDOW_LENGTH = 30
STEP_SIZE = 5
CORE_JOINTS = [5, 6, 11, 12]

# Reported FPS verified using check_video_fps.py.
FPS_BY_DATASET = {
    "Coffee_room_01": 25,
    "Coffee_room_02": 25,
    "Home_01": 24,
    "Home_02": 24,
}

MAX_GAP_SECONDS = 0.12
MAX_MISSING_FRACTION = 0.30


def remaining_missing_after_short_gaps(missing, max_gap_frames):
    """
    Estimate missingness after filling short, bounded gaps.

    Works on each joint separately, inside this window only.
    Does not modify or save the original keypoints.
    """
    remaining = missing.copy()
    frame_count, joint_count = missing.shape

    for joint in range(joint_count):
        position = 0

        while position < frame_count:
            if not missing[position, joint]:
                position += 1
                continue

            start = position

            while position < frame_count and missing[position, joint]:
                position += 1

            end = position  # First frame after the gap.
            gap_length = end - start

            # Valid points must exist on both sides.
            bounded = start > 0 and end < frame_count

            if bounded and gap_length <= max_gap_frames:
                remaining[start:end, joint] = False

    return remaining


def main():
    files = sorted(POSES_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No pose files found: {POSES_DIR}")

    baseline_count = 0
    strict_count = 0
    flexible_count = 0

    fall_videos = set()
    baseline_covered = set()
    strict_covered = set()
    flexible_covered = set()
    seen_ids = set()

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            video_id = str(data["video_id"].item())
            keypoints = data["keypoints"]
            frame_numbers = data["frame_numbers"]
            fall_start = int(data["fall_start"].item())
            fall_end = int(data["fall_end"].item())

        if video_id in seen_ids:
            raise ValueError(f"Duplicate video ID: {video_id}")

        seen_ids.add(video_id)

        # Get the correct FPS for this dataset.
        dataset_name = video_id.split("/")[0]

        if dataset_name not in FPS_BY_DATASET:
            raise ValueError(
                f"Unknown dataset for FPS lookup: {dataset_name}"
            )

        fps = FPS_BY_DATASET[dataset_name]

        max_gap_frames = math.floor(
            fps * MAX_GAP_SECONDS + 1e-9
        )

        if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
            raise ValueError(
                f"Unexpected keypoint shape: {video_id}"
            )

        frame_count = len(keypoints)

        if not np.array_equal(
            frame_numbers,
            np.arange(1, frame_count + 1)
        ):
            raise ValueError(
                f"Unexpected frame numbering: {video_id}"
            )

        # This comparison measures Fall-window retention only.
        if (fall_start, fall_end) == (0, 0):
            continue

        if not 1 <= fall_start <= fall_end <= frame_count:
            raise ValueError(
                f"Invalid fall interval: {video_id}"
            )

        fall_videos.add(video_id)
        fall_length = fall_end - fall_start + 1

        for start in range(
            0,
            frame_count - WINDOW_LENGTH + 1,
            STEP_SIZE
        ):
            # Annotation numbering starts at 1.
            window_start = start + 1
            window_end = window_start + WINDOW_LENGTH - 1

            # Exclude only windows entirely after the fall.
            # Windows containing the fall and landing are allowed.
            if window_start > fall_end:
                continue

            # Also exclude windows ending over 1 second after fall_end.
            max_postfall_frames = math.floor(fps * 1.0 + 1e-9)

            if window_end > fall_end + max_postfall_frames:
                continue

            overlap = max(
                0,
                min(window_end, fall_end)
                - max(window_start, fall_start)
                + 1
            )

            # Same adaptive Fall rule used in the earlier experiment.
            required_overlap = (
                0.5 * min(WINDOW_LENGTH, fall_length)
            )

            if overlap < required_overlap:
                continue

            baseline_count += 1
            baseline_covered.add(video_id)

            window = keypoints[
                start:start + WINDOW_LENGTH
            ]

            # A joint is missing if either x or y is not finite.
            missing = ~np.isfinite(
                window[:, CORE_JOINTS, :2]
            ).all(axis=2)

            remaining = remaining_missing_after_short_gaps(
                missing,
                max_gap_frames
            )

            missing_fraction = remaining.mean()

            # Strict rule:
            # Keep only windows with no remaining missing
            # core-joint observations.
            if not remaining.any():
                strict_count += 1
                strict_covered.add(video_id)

            # Flexible rule:
            # Keep windows with at most 30% remaining missing data.
            if missing_fraction <= MAX_MISSING_FRACTION:
                flexible_count += 1
                flexible_covered.add(video_id)

    print("Pose files checked:", len(files))
    print("Fall videos:", len(fall_videos))

    print(
        "Verified FPS by dataset:",
        FPS_BY_DATASET
    )

    print(
        "Maximum short gap:",
        MAX_GAP_SECONDS,
        "seconds"
    )

    print(
        "Gap limits: Home = 2 frames; "
        "Coffee rooms = 3 frames"
    )

    print(
        "Baseline Fall windows:",
        baseline_count
    )

    results = [
        (
            "Before quality filtering",
            baseline_count,
            baseline_covered
        ),
        (
            "Strict: no remaining missing core joints",
            strict_count,
            strict_covered
        ),
        (
            "Flexible: at most 30% missing core joints",
            flexible_count,
            flexible_covered
        ),
    ]

    for name, kept, covered in results:
        removed = baseline_count - kept

        percentage = (
            100 * removed / baseline_count
            if baseline_count
            else 0
        )

        print(f"\n--- {name} ---")
        print(
            "Fall windows kept:",
            kept
        )

        print(
            f"Fall windows removed: "
            f"{removed} ({percentage:.1f}%)"
        )

        print(
            f"Falls covered: "
            f"{len(covered)}/{len(fall_videos)}"
        )

        uncovered = sorted(
            fall_videos - covered
        )

        if uncovered:
            print(
                "Falls without a retained Fall window:"
            )

            for video_id in uncovered:
                print(" ", video_id)


if __name__ == "__main__":
    main()