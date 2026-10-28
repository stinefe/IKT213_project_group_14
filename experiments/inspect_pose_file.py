from pathlib import Path
import numpy as np

from falldet.config import OUTPUTS_DIR
POSES_DIR = OUTPUTS_DIR / "poses"

files = sorted(POSES_DIR.glob("*.npz"))

if not files:
    raise FileNotFoundError(f"No pose files found in: {POSES_DIR}")

# Inspect one file without changing it.
pose_file = files[0]

print("File:", pose_file.name)

with np.load(pose_file, allow_pickle=False) as data:
    print("Available fields:", data.files)

    for name in data.files:
        values = data[name]
        print(f"{name}: shape={values.shape}, type={values.dtype}")

        # Print small metadata values, not the full keypoints array.
        if values.size == 1:
            print("  Value:", values.item())