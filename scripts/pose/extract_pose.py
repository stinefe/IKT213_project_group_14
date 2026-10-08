from falldet.config import (INDEX_FILE, OUTPUTS_DIR, MODEL_PATH, POSE_VIDEO_ID,
                    POSE_OUTPUT_NAME, require_dataset_root, resolve_dataset_path)
import csv
import numpy as np
from ultralytics import YOLO

#1# Find the video in our dataset index
require_dataset_root()
index_file = INDEX_FILE
video_id = POSE_VIDEO_ID

with open(index_file, "r", encoding="utf-8-sig", newline="") as file:
    records = list(csv.DictReader(file))

record = next((row for row in records if row["video_id"] == video_id), None)
if record is None:
    raise ValueError(f"Video ID not found in dataset_index.csv: {video_id}")
video = resolve_dataset_path(record["video_path"])
expected_frames = int(record["frame_count"])

if not video.is_file():
    raise FileNotFoundError(video)

#2# Load the pretrained pose model
model = YOLO(str(MODEL_PATH))

output_folder = OUTPUTS_DIR / POSE_OUTPUT_NAME
output_folder.mkdir(parents=True, exist_ok=True)

#3# Process every frame and save a skeleton preview video
results = model.predict(
    source=str(video),
    stream=True,
    device="cpu",
    imgsz=640,
    conf=0.25,
    vid_stride=1,
    save=True,
    project=str(output_folder.parent),
    name=output_folder.name,
    exist_ok=True,
    verbose=False,
)

all_keypoints = []
person_counts = []

#4# Collect 17 keypoints for each frame
for frame_number, result in enumerate(results, start=1):
    count = len(result.boxes) if result.boxes is not None else 0
    person_counts.append(count)

    # Each keypoint contains: normalized x, normalized y, confidence
    points = np.full((17, 3), np.nan, dtype=np.float32)

    # For this first test, only accept frames with exactly one person
    if count == 1 and result.keypoints is not None:
        coordinates = result.keypoints.xyn[0].cpu().numpy().copy()
        confidence = result.keypoints.conf[0].cpu().numpy()

        # Mark uncertain coordinates as missing
        coordinates[confidence < 0.5] = np.nan

        points[:, :2] = coordinates
        points[:, 2] = confidence

    # Preserve the frame even when its pose is missing or ambiguous
    all_keypoints.append(points)

    if frame_number % 25 == 0:
        print("Frames processed:", frame_number)

#5# Verify that no frames were lost
processed = len(all_keypoints)

if processed != expected_frames:
    raise ValueError(
        f"Frame mismatch: processed {processed}, expected {expected_frames}"
    )

keypoints = np.stack(all_keypoints)
person_counts = np.asarray(person_counts)

#6# Save the numerical keypoints
output_file = output_folder / "keypoints.npz"

np.savez_compressed(
    output_file,
    video_id=video_id,
    frame_numbers=np.arange(1, processed + 1),
    keypoints=keypoints,
    person_counts=person_counts,
    keypoint_confidence_threshold=0.5,
)

print("Frames processed:", processed)
print("Frames with one person:", int(np.sum(person_counts == 1)))
print("Frames with no person:", int(np.sum(person_counts == 0)))
print("Frames with multiple people:", int(np.sum(person_counts > 1)))
print("Keypoints shape:", keypoints.shape)
print("Keypoints saved:", output_file)
print("Preview folder:", output_folder)
