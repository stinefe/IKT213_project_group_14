from config import DATASET_ROOT, INDEX_FILE, ANNOTATION_FOLDERS, require_dataset_root
import csv
import cv2

#1# Shared dataset location and annotation folders
require_dataset_root()
root = DATASET_ROOT
datasets = ANNOTATION_FOLDERS

# These two videos were reviewed using sampled frames only
reviewed_samples = {
    ("Home_02", "video (38)"),
    ("Coffee_room_02", "video (61)"),
}

records = []

#3# Collect and validate each video's information
for name, annotation_folder in datasets.items():
    folder = root / name / name
    videos_folder = folder / "Videos"
    labels_folder = folder / annotation_folder

    if not videos_folder.is_dir() or not labels_folder.is_dir():
        raise ValueError(f"Missing folder in {name}")

    included = 0

    for video in sorted(videos_folder.glob("*.avi")):
        # Exclude only the known problematic Coffee_room_01 pair
        if name == "Coffee_room_01" and video.stem == "video (35)":
            continue

        label = labels_folder / f"{video.stem}.txt"

        if not label.is_file():
            raise ValueError(f"Missing annotation: {label}")

        fall_values = []
        frame_numbers = []

        with open(label, "r", encoding="utf-8-sig") as file:
            for line in file:
                if not line.strip():
                    continue

                columns = line.strip().split(",")

                if len(columns) == 1:
                    fall_values.append(int(columns[0]))
                elif len(columns) == 6:
                    frame_numbers.append(int(columns[0]))
                else:
                    raise ValueError(f"Unexpected row in {label}: {line}")

        cap = cv2.VideoCapture(str(video))
        opened = cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if not opened or frame_count <= 0:
            raise ValueError(f"Cannot read video information: {video}")

        if frame_numbers != list(range(1, frame_count + 1)):
            raise ValueError(f"Annotation count/order problem: {label}")

        if len(fall_values) != 2:
            raise ValueError(f"Expected two fall values: {label}")

        start, end = fall_values

        if start == 0 and end == 0:
            annotation_status = "zero_zero"
        elif 1 <= start <= end <= frame_count:
            annotation_status = "fall_interval"
        else:
            raise ValueError(f"Invalid fall interval: {label}")

        review_status = (
            "sampled_frames_no_fall_seen"
            if (name, video.stem) in reviewed_samples
            else "not_recorded_in_this_index"
        )

        records.append({
            "video_id": f"{name}/{video.stem}",
            "dataset": name,
            "video_path": video.relative_to(root).as_posix(),
            "annotation_path": label.relative_to(root).as_posix(),
            "frame_count": frame_count,
            "fall_start": start,
            "fall_end": end,
            "annotation_status": annotation_status,
            "visual_review": review_status,
        })

        included += 1

    print(name, "| Included videos:", included)

#4# Check the total before writing the table
if len(records) != 129:
    raise ValueError(f"Expected 129 videos, found {len(records)}")

#5# Save beside this Python script
output = INDEX_FILE

with open(output, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

print("Total indexed:", len(records))
print("With fall interval:",
      sum(row["annotation_status"] == "fall_interval" for row in records))
print("With zero-zero values:",
      sum(row["annotation_status"] == "zero_zero" for row in records))
print("Saved:", output)
