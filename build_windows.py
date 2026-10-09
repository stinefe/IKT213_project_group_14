"""Build labelled keypoint windows for classifier training."""

import numpy as np

from falldet.config import OUTPUTS_DIR
from pose_preprocessing import interpolate_short_gaps

POSES_DIR = OUTPUTS_DIR / "poses"
OUTPUT_FILE = OUTPUTS_DIR / "windows.npz"

WINDOW_LENGTH = 30
STEP_SIZE = 5
CORE_JOINTS = [5, 6, 11, 12]

MAX_GAP_SECONDS = 0.12
MAX_MISSING_FRACTION = 0.30

# Verified using check_video_fps.py.
FPS_BY_DATASET = {
    "Coffee_room_01": 25,
    "Coffee_room_02": 25,
    "Home_01": 24,
    "Home_02": 24,
}

# These zero-zero videos were manually reviewed.
REVIEWED_NORMAL_IDS = {
    f"Home_02/video ({number})"
    for number in range(38, 61)
} | {
    f"Coffee_room_02/video ({number})"
    for number in [61, 63, 65, 66, 67, 68, 69, 70]
}


def choose_label(window_start, window_end, fall_start, fall_end):
    """Return 0=Normal, 1=Fall, or None=exclude."""

    # Manually reviewed no-fall video.
    if (fall_start, fall_end) == (0, 0):
        return 0

    # Entire window is before the fall.
    if window_end < fall_start:
        return 0

    # Entire window is after the fall.
    if window_start > fall_end:
        return None

    fall_length = fall_end - fall_start + 1

    overlap = max(
        0,
        min(window_end, fall_end)
        - max(window_start, fall_start)
        + 1,
    )

    # Tested adaptive overlap rule.
    required_overlap = 0.5 * min(WINDOW_LENGTH, fall_length)

    if overlap >= required_overlap:
        return 1

    # A partly overlapping window that fails the Fall rule
    # must not automatically become Normal.
    return None


def main():
    files = sorted(POSES_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No pose files found: {POSES_DIR}"
        )

    windows = []
    labels = []
    video_ids = []
    start_frames = []
    end_frames = []
    fps_values = []
    missing_fractions = []
    interpolation_masks = []

    seen_ids = set()
    fall_videos = set()
    represented_falls = set()

    # Index 0 = Normal; index 1 = Fall.
    candidates = [0, 0]
    removed = [0, 0]
    excluded_by_label = 0

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

        dataset_name = video_id.split("/")[0]

        if dataset_name not in FPS_BY_DATASET:
            raise ValueError(
                f"Unknown dataset: {dataset_name}"
            )

        fps = FPS_BY_DATASET[dataset_name]
        no_fall = (fall_start, fall_end) == (0, 0)

        if no_fall:
            if video_id not in REVIEWED_NORMAL_IDS:
                raise ValueError(
                    f"Zero-zero video needs manual review: {video_id}"
                )
        else:
            if not 1 <= fall_start <= fall_end <= frame_count:
                raise ValueError(
                    f"Invalid fall interval: {video_id}"
                )

            fall_videos.add(video_id)

        for start in range(
            0,
            frame_count - WINDOW_LENGTH + 1,
            STEP_SIZE,
        ):
            window_start = start + 1
            window_end = window_start + WINDOW_LENGTH - 1

            label = choose_label(
                window_start,
                window_end,
                fall_start,
                fall_end,
            )

            if label is None:
                excluded_by_label += 1
                continue

            candidates[label] += 1

            window = keypoints[
                start:start + WINDOW_LENGTH
            ]

            processed, interpolated, remaining = (
                interpolate_short_gaps(
                    window,
                    fps,
                    max_gap_seconds=MAX_GAP_SECONDS,
                )
            )

            # Quality is measured over 30 frames × 4 core joints.
            missing_fraction = float(
                remaining[:, CORE_JOINTS].mean()
            )

            if missing_fraction > MAX_MISSING_FRACTION:
                removed[label] += 1
                continue

            # Represent every remaining missing joint's x and y
            # as NaN, including any originally infinite values.
            xy = processed[:, :, :2]
            xy[remaining] = np.nan

            # Preserve confidence values and the coordinate scale.
            windows.append(processed.astype(np.float32))
            labels.append(label)
            video_ids.append(video_id)
            start_frames.append(window_start)
            end_frames.append(window_end)
            fps_values.append(fps)
            missing_fractions.append(missing_fraction)
            interpolation_masks.append(interpolated)

            if label == 1:
                represented_falls.add(video_id)

    if not windows:
        raise ValueError("No windows passed the rules.")

    X = np.stack(windows)
    y = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(video_ids, dtype=str)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,

        # Required arrays.
        X=X,
        y=y,
        video_id=ids,

        # Extra information for tracing and checking windows.
        window_start=np.asarray(start_frames, dtype=np.int64),
        window_end=np.asarray(end_frames, dtype=np.int64),
        fps=np.asarray(fps_values, dtype=np.float32),
        core_missing_fraction=np.asarray(
            missing_fractions, dtype=np.float32
        ),
        interpolated_mask=np.stack(interpolation_masks),

        # Record the settings used to build this file.
        window_length=WINDOW_LENGTH,
        step_size=STEP_SIZE,
        max_gap_seconds=MAX_GAP_SECONDS,
        max_missing_fraction=MAX_MISSING_FRACTION,
        core_joints=np.asarray(CORE_JOINTS, dtype=np.int64),
        fall_overlap_rule="overlap >= 0.5 * min(window_length, fall_length)",
        postfall_rule="exclude windows starting after fall_end",
    )

    # Reopen the saved file and verify its contents.
    with np.load(OUTPUT_FILE, allow_pickle=False) as saved:
        if not np.array_equal(saved["X"], X, equal_nan=True):
            raise AssertionError("Saved keypoints differ from memory.")

        if not np.array_equal(saved["y"], y):
            raise AssertionError("Saved labels differ from memory.")

        if not np.array_equal(saved["video_id"], ids):
            raise AssertionError("Saved video IDs differ from memory.")

        actual_missing = ~np.isfinite(
            saved["X"][:, :, CORE_JOINTS, :2]
        ).all(axis=-1)

        saved_quality = actual_missing.mean(axis=(1, 2))

        if np.any(saved_quality > MAX_MISSING_FRACTION):
            raise AssertionError(
                "A saved window fails the quality rule."
            )

    print("\n--- WINDOW DATASET SUMMARY ---")
    print("Pose files checked:", len(files))
    print("Window length:", WINDOW_LENGTH, "frames")
    print("Step size:", STEP_SIZE, "frames")

    for label, name in [(0, "Normal"), (1, "Fall")]:
        kept = int(np.count_nonzero(y == label))
        print(
            f"{name}: candidates={candidates[label]} "
            f"| kept={kept} "
            f"| removed by quality={removed[label]}"
        )

    print("Windows excluded by labelling rules:", excluded_by_label)
    print("Total windows saved:", len(y))
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("video_id shape:", ids.shape)
    print(
        "Falls represented:",
        f"{len(represented_falls)}/{len(fall_videos)}",
    )

    for video_id in sorted(fall_videos - represented_falls):
        print("NO RETAINED FALL WINDOW:", video_id)

    print("Saved-file verification: PASS")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()