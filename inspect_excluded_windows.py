import numpy as np

from test_postfall_step5 import (
    POSES_DIR,
    WINDOW_LENGTH,
    STEP_SIZE,
    CORE_JOINTS,
    MAX_MISSING_FRACTION,
    remaining_missing_after_short_gaps,
)

TARGET_VIDEOS = {
    "Home_01/video (11)",
    "Home_02/video (35)",
}

JOINT_NAMES = {
    5: "Left shoulder",
    6: "Right shoulder",
    11: "Left hip",
    12: "Right hip",
}

found = set()

for path in sorted(POSES_DIR.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
        video_id = str(data["video_id"].item())

        if video_id not in TARGET_VIDEOS:
            continue

        keypoints = data["keypoints"]
        fall_start = int(data["fall_start"].item())
        fall_end = int(data["fall_end"].item())

    found.add(video_id)
    fall_length = fall_end - fall_start + 1
    required_overlap = 0.5 * min(WINDOW_LENGTH, fall_length)

    print(f"\nVIDEO: {video_id}")
    print(f"Fall interval: {fall_start}–{fall_end}")

    for start in range(
        0, len(keypoints) - WINDOW_LENGTH + 1, STEP_SIZE
    ):
        window_start = start + 1
        window_end = window_start + WINDOW_LENGTH - 1

        overlap = max(
            0,
            min(window_end, fall_end)
            - max(window_start, fall_start)
            + 1
        )
        if window_end > fall_end:
            continue

        if overlap < required_overlap:
            continue

        window = keypoints[start:start + WINDOW_LENGTH]

        missing = ~np.isfinite(
            window[:, CORE_JOINTS, :2]
        ).all(axis=2)

        remaining = remaining_missing_after_short_gaps(missing)
        fraction = remaining.mean()

        decision = (
            "KEEP"
            if fraction <= MAX_MISSING_FRACTION
            else "EXCLUDE"
        )

        print(f"\n  Window: {window_start}–{window_end}")
        print(f"  Frames overlapping the fall: {overlap}")
        print(f"  Missing before short-gap fill: {missing.mean():.1%}")
        print(f"  Missing after estimated fill: {fraction:.1%}")
        print(f"  Decision at 30%: {decision}")

        for column, joint in enumerate(CORE_JOINTS):
            count = int(remaining[:, column].sum())
            print(
                f"    {JOINT_NAMES[joint]}: "
                f"missing in {count}/{WINDOW_LENGTH} frames"
            )

        after_fall = max(0, window_end - fall_end)
        print(f"  Frames after fall_end: {after_fall}")

for video_id in sorted(TARGET_VIDEOS - found):
    print("POSE FILE NOT FOUND:", video_id)