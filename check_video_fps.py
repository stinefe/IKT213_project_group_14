import csv
import math
from collections import Counter

import cv2

from falldet.config import INDEX_FILE, require_dataset_root, resolve_dataset_path

require_dataset_root()

fps_counts = Counter()
problems = []

with INDEX_FILE.open("r", encoding="utf-8-sig", newline="") as file:
    videos = list(csv.DictReader(file))

for row in videos:
    video_id = row["video_id"]
    video_path = resolve_dataset_path(row["video_path"])

    if not video_path.is_file():
        problems.append(f"{video_id}: video file not found")
        continue

    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            problems.append(f"{video_id}: cannot open video")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()

    if not math.isfinite(fps) or fps <= 0:
        problems.append(f"{video_id}: invalid FPS ({fps})")
        continue

    fps_counts[round(fps, 3)] += 1

    print(f"{video_id} | FPS: {fps:.3f}")

print("\n--- FPS SUMMARY ---")
print("Videos in index:", len(videos))
print("Videos with valid FPS:", sum(fps_counts.values()))

for fps, count in sorted(fps_counts.items()):
    print(f"{fps:.3f} FPS: {count} videos")

print("Problems:", len(problems))

for problem in problems:
    print("CHECK:", problem)