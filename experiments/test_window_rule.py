import csv
from falldet.config import INDEX_FILE

WINDOW_LENGTH = 30
STEP_SIZES = [5, 10, 15]


def main():
    # Read the dataset index using the path in config.py.
    with INDEX_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        videos = list(csv.DictReader(file))

    # Check frame counts and fall intervals before testing.
    for video in videos:
        frames = int(video["frame_count"])
        start = int(video["fall_start"])
        end = int(video["fall_end"])

        if frames <= 0:
            raise ValueError(
                f"Invalid frame count: {video['video_id']}"
            )

        if (start, end) != (0, 0) and not (
            1 <= start <= end <= frames
        ):
            raise ValueError(
                f"Invalid fall interval: {video['video_id']}"
            )

    for step in STEP_SIZES:
        total_windows = 0
        fall_windows = 0
        total_falls = 0
        covered_falls = 0
        missed_falls = []

        for video in videos:
            frames = int(video["frame_count"])
            fall_start = int(video["fall_start"])
            fall_end = int(video["fall_end"])

            has_fall = (fall_start, fall_end) != (0, 0)
            found_fall_window = False

            if has_fall:
                total_falls += 1
                fall_length = fall_end - fall_start + 1

            # Full 30-frame windows only; no padding at the end.
            for window_start in range(
                1, frames - WINDOW_LENGTH + 2, step
            ):
                window_end = window_start + WINDOW_LENGTH - 1
                total_windows += 1

                # Zero-zero videos have no fall interval to overlap.
                if not has_fall:
                    continue

                # Count frames shared by the window and the fall.
                overlap = max(
                    0,
                    min(window_end, fall_end)
                    - max(window_start, fall_start)
                    + 1,
                )

                # Your friend's proposed rule.
                required_overlap = 0.5 * min(
                    WINDOW_LENGTH, fall_length
                )

                if overlap >= required_overlap:
                    fall_windows += 1
                    found_fall_window = True

            if has_fall:
                if found_fall_window:
                    covered_falls += 1
                else:
                    missed_falls.append(video["video_id"])

        print(f"\n--- Step size: {step} frames ---")
        print("Total windows:", total_windows)
        print("Windows meeting the Fall rule:", fall_windows)
        print(
            "Falls with at least one Fall window:",
            f"{covered_falls}/{total_falls}",
        )
        print("Falls without a Fall window:", len(missed_falls))

        for video_id in missed_falls:
            print("  CHECK:", video_id)


if __name__ == "__main__":
    main()