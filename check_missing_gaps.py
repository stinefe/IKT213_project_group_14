from pathlib import Path
import numpy as np
from collections import Counter

POSES_DIR = Path("outputs/poses")
CORE_JOINTS = [5, 6, 11, 12]

gap_lengths = []

for npz_file in POSES_DIR.glob("*.npz"):
    data = np.load(npz_file)
    keypoints = data["keypoints"]

    # Frame is considered missing if any core joint is missing
    missing = np.isnan(keypoints[:, CORE_JOINTS, 0]).any(axis=1)

    current_gap = 0

    for is_missing in missing:
        if is_missing:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_lengths.append(current_gap)
                current_gap = 0

    if current_gap > 0:
        gap_lengths.append(current_gap)

counts = Counter(gap_lengths)

print("Total missing gaps:", len(gap_lengths))

for length in sorted(counts):
    print(f"{length} frame gap: {counts[length]}")

if gap_lengths:
    print("\nLongest gap:", max(gap_lengths), "frames")