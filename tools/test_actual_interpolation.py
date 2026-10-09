import math
import numpy as np

from falldet.config import OUTPUTS_DIR
from falldet.pose_preprocessing import interpolate_short_gaps
from experiments.test_postfall_actual_fps import (
    WINDOW_LENGTH,
    STEP_SIZE,
    FPS_BY_DATASET,
    MAX_GAP_SECONDS,
    remaining_missing_after_short_gaps,
)

POSES_DIR = OUTPUTS_DIR / "poses"


def main():
    files = sorted(POSES_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No pose files found: {POSES_DIR}"
        )

    windows_checked = 0
    filled_observations = 0
    seen_ids = set()

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            video_id = str(data["video_id"].item())
            keypoints = data["keypoints"]
            frame_numbers = data["frame_numbers"]

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

        if not np.array_equal(
            frame_numbers,
            np.arange(1, len(keypoints) + 1),
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
        max_gap_frames = math.floor(
            fps * MAX_GAP_SECONDS + 1e-9
        )

        # Check every complete window, independently of its label.
        for start in range(
            0,
            len(keypoints) - WINDOW_LENGTH + 1,
            STEP_SIZE,
        ):
            window = keypoints[
                start:start + WINDOW_LENGTH
            ].copy()

            original = window.copy()
            location = (
                f"{video_id}, frames "
                f"{start + 1}–{start + WINDOW_LENGTH}"
            )

            # Check all 17 joints.
            missing_before = ~np.isfinite(
                original[:, :, :2]
            ).all(axis=2)

            expected_remaining = (
                remaining_missing_after_short_gaps(
                    missing_before,
                    max_gap_frames,
                )
            )

            processed, filled, remaining = (
                interpolate_short_gaps(
                    window,
                    fps,
                    max_gap_seconds=MAX_GAP_SECONDS,
                )
            )

            # 1. The input must remain unchanged.
            if not np.array_equal(
                window, original, equal_nan=True
            ):
                raise AssertionError(
                    f"Input was modified: {location}"
                )

            # 2. Output shapes must be correct.
            if (
                processed.shape != original.shape
                or filled.shape != missing_before.shape
                or remaining.shape != missing_before.shape
            ):
                raise AssertionError(
                    f"Wrong output shape: {location}"
                )

            # 3. Actual missing coordinates must match the estimate.
            actual_remaining = ~np.isfinite(
                processed[:, :, :2]
            ).all(axis=2)

            if not np.array_equal(
                actual_remaining, expected_remaining
            ):
                raise AssertionError(
                    f"Missing-data estimate differs: {location}"
                )

            if not np.array_equal(
                remaining, actual_remaining
            ):
                raise AssertionError(
                    f"Incorrect remaining mask: {location}"
                )

            # 4. Only eligible missing observations may be filled.
            expected_filled = (
                missing_before & ~expected_remaining
            )

            if not np.array_equal(filled, expected_filled):
                raise AssertionError(
                    f"Incorrect interpolation mask: {location}"
                )

            # Coordinates outside the filled observations stay unchanged.
            if not np.array_equal(
                processed[:, :, :2][~expected_filled],
                original[:, :, :2][~expected_filled],
                equal_nan=True,
            ):
                raise AssertionError(
                    f"Other coordinates changed: {location}"
                )

            # 5. Confidence values must remain unchanged.
            if not np.array_equal(
                processed[:, :, 2],
                original[:, :, 2],
                equal_nan=True,
            ):
                raise AssertionError(
                    f"Confidence values changed: {location}"
                )

            # 6. Filled coordinates must follow linear interpolation.
            for joint in range(17):
                targets = np.flatnonzero(
                    expected_filled[:, joint]
                )

                if len(targets) == 0:
                    continue

                valid_frames = np.flatnonzero(
                    ~missing_before[:, joint]
                )

                for coordinate in (0, 1):
                    expected_values = np.interp(
                        targets,
                        valid_frames,
                        original[
                            valid_frames, joint, coordinate
                        ],
                    )

                    if not np.allclose(
                        processed[targets, joint, coordinate],
                        expected_values,
                        rtol=1e-5,
                        atol=1e-6,
                    ):
                        raise AssertionError(
                            f"Incorrect filled coordinates: {location}"
                        )

            windows_checked += 1
            filled_observations += int(
                expected_filled.sum()
            )

    if windows_checked == 0:
        raise ValueError("No complete windows were checked.")

    print("\n--- ACTUAL INTERPOLATION CHECK ---")
    print("Pose files checked:", len(files))
    print("Windows checked:", windows_checked)
    print(
        "Joint observations interpolated across windows:",
        filled_observations,
    )
    print("Missing-data masks match estimates: PASS")
    print("Filled coordinates match linear interpolation: PASS")
    print("Original data and confidence values unchanged: PASS")
    print("No files were saved or overwritten.")


if __name__ == "__main__":
    main()