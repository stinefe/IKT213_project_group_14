import math
import numpy as np

from config import OUTPUTS_DIR
from test_postfall_actual_fps import (
    WINDOW_LENGTH,
    STEP_SIZE,
    CORE_JOINTS,
    FPS_BY_DATASET,
    MAX_GAP_SECONDS,
    MAX_MISSING_FRACTION,
    remaining_missing_after_short_gaps,
)

POSES_DIR = OUTPUTS_DIR / "poses"
EXPECTED_FALL_VIDEOS = 98


def main():
    total_windows = 0
    kept_windows = 0

    fall_videos = set()
    covered_videos = set()
    seen_ids = set()

    files = sorted(POSES_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No pose files found: {POSES_DIR}"
        )

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            video_id = str(data["video_id"].item())
            fall_start = int(data["fall_start"].item())
            fall_end = int(data["fall_end"].item())
            keypoints = data["keypoints"]
            frame_numbers = data["frame_numbers"]

        if video_id in seen_ids:
            raise ValueError(
                f"Duplicate video ID: {video_id}"
            )

        seen_ids.add(video_id)

        # Zero-zero videos are checked in the separate Normal test.
        if (fall_start, fall_end) == (0, 0):
            continue

        if (
            keypoints.ndim != 3
            or keypoints.shape[1:] != (17, 3)
        ):
            raise ValueError(
                f"Unexpected keypoint shape: {video_id}"
            )

        frame_count = len(keypoints)

        if not np.array_equal(
            frame_numbers,
            np.arange(1, frame_count + 1),
        ):
            raise ValueError(
                f"Unexpected frame numbering: {video_id}"
            )

        if not 1 <= fall_start <= fall_end <= frame_count:
            raise ValueError(
                f"Invalid fall interval: {video_id}"
            )

        fall_videos.add(video_id)

        dataset_name = video_id.split("/")[0]

        if dataset_name not in FPS_BY_DATASET:
            raise ValueError(
                f"Unknown dataset for FPS lookup: {dataset_name}"
            )

        fps = FPS_BY_DATASET[dataset_name]
        max_gap_frames = math.floor(
            fps * MAX_GAP_SECONDS + 1e-9
        )

        video_total = 0
        video_kept = 0

        for start in range(
            0,
            frame_count - WINDOW_LENGTH + 1,
            STEP_SIZE,
        ):
            # Array positions start at 0; annotation frames start at 1.
            window_start = start + 1
            window_end = window_start + WINDOW_LENGTH - 1

            # Normal windows must end BEFORE the first fall frame.
            # Later windows cannot satisfy this, so stop this loop.
            if window_end >= fall_start:
                break

            window = keypoints[
                start:start + WINDOW_LENGTH
            ]

            # A core joint is missing if either x or y is not finite.
            missing = ~np.isfinite(
                window[:, CORE_JOINTS, :2]
            ).all(axis=2)

            # Estimate the effect of filling short, bounded gaps.
            # This does not interpolate actual coordinates.
            remaining = remaining_missing_after_short_gaps(
                missing,
                max_gap_frames,
            )

            video_total += 1

            # Fraction of missing core-joint observations.
            if remaining.mean() <= MAX_MISSING_FRACTION:
                video_kept += 1

        total_windows += video_total
        kept_windows += video_kept

        if video_kept > 0:
            covered_videos.add(video_id)

        print(
            f"{video_id} | Total: {video_total} "
            f"| Kept: {video_kept} "
            f"| Removed: {video_total - video_kept}"
        )

    if len(fall_videos) != EXPECTED_FALL_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_FALL_VIDEOS} fall videos, "
            f"found {len(fall_videos)}."
        )

    removed = total_windows - kept_windows
    removed_percent = (
        100 * removed / total_windows
        if total_windows
        else 0
    )

    print("\n--- PRE-FALL NORMAL WINDOW SUMMARY ---")
    print("Fall videos checked:", len(fall_videos))
    print("Window length:", WINDOW_LENGTH, "frames")
    print("Step size:", STEP_SIZE, "frames")
    print("Normal windows before filtering:", total_windows)
    print("Normal windows kept:", kept_windows)
    print(
        f"Normal windows removed: "
        f"{removed} ({removed_percent:.1f}%)"
    )
    print(
        "Videos with at least one retained Normal window:",
        f"{len(covered_videos)}/{len(fall_videos)}",
    )

    for video_id in sorted(fall_videos - covered_videos):
        print("NO RETAINED PRE-FALL WINDOWS:", video_id)


if __name__ == "__main__":
    main()