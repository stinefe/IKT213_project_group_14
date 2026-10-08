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

total_windows = 0
kept_windows = 0
normal_videos = set()
covered_videos = set()

files = sorted(POSES_DIR.glob("*.npz"))

if not files:
    raise FileNotFoundError(f"No pose files found: {POSES_DIR}")

for path in files:
    with np.load(path, allow_pickle=False) as data:
        fall_start = int(data["fall_start"].item())
        fall_end = int(data["fall_end"].item())

        # Only check the manually reviewed zero-zero videos.
        if (fall_start, fall_end) != (0, 0):
            continue

        video_id = str(data["video_id"].item())
        keypoints = data["keypoints"]
        frame_numbers = data["frame_numbers"]

    if video_id in normal_videos:
        raise ValueError(f"Duplicate video ID: {video_id}")

    normal_videos.add(video_id)

    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"Unexpected keypoint shape: {video_id}")

    if not np.array_equal(
        frame_numbers, np.arange(1, len(keypoints) + 1)
    ):
        raise ValueError(f"Unexpected frame numbering: {video_id}")

    dataset_name = video_id.split("/")[0]
    fps = FPS_BY_DATASET[dataset_name]
    max_gap_frames = math.floor(fps * MAX_GAP_SECONDS + 1e-9)

    video_total = 0
    video_kept = 0

    # No post-fall cutoff: these videos contain no annotated fall.
    for start in range(
        0, len(keypoints) - WINDOW_LENGTH + 1, STEP_SIZE
    ):
        window = keypoints[start:start + WINDOW_LENGTH]

        missing = ~np.isfinite(
            window[:, CORE_JOINTS, :2]
        ).all(axis=2)

        remaining = remaining_missing_after_short_gaps(
            missing, max_gap_frames
        )

        video_total += 1

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

if len(normal_videos) != 31:
    raise ValueError(
        f"Expected 31 zero-zero videos, found {len(normal_videos)}."
    )

removed = total_windows - kept_windows
removed_percent = (
    100 * removed / total_windows if total_windows else 0
)

print("\n--- NORMAL WINDOW SUMMARY ---")
print("Zero-zero videos checked:", len(normal_videos))
print("Normal windows before filtering:", total_windows)
print("Normal windows kept:", kept_windows)
print(f"Normal windows removed: {removed} ({removed_percent:.1f}%)")
print(
    "Videos with at least one retained Normal window:",
    f"{len(covered_videos)}/{len(normal_videos)}"
)

for video_id in sorted(normal_videos - covered_videos):
    print("NO RETAINED WINDOWS:", video_id)