from pathlib import Path
import cv2

#1# Location of Coffee_room_01
dataset = Path(
    r"C:\Users\haifa\Desktop\FallDataset\Home_02\Home_02"
)

#2# creates the path to the Videos and annotation folder
videos_folder = dataset / "Videos"
labels_folder = dataset / "Annotation_files_cleaned"

print("RUNNING FILE:", __file__)
print("VIDEO PATH:", repr(str(videos_folder)))
print("LABEL PATH:", repr(str(labels_folder)))

#3# check whether the folders really exist on our computer.
print("Videos folder found:", videos_folder.is_dir())
print("Labels folder found:", labels_folder.is_dir())


#4# count how many video files and label files you have
print("Number of videos:", len(list(videos_folder.glob("*.avi"))))
print("Number of labels:", len(list(labels_folder.glob("*.txt"))))

#5# Check that every video has a label file with the same name
for video in videos_folder.glob("*.avi"):
    label = labels_folder / (video.stem + ".txt")

    if not label.is_file():
        print("Missing label for:", video.name)

for label in labels_folder.glob("*.txt"):
    video = videos_folder / (label.stem + ".avi")

    if not video.is_file():
        print("Missing video for:", label.name)

print("File matching check finished.")


#6# Check whether Python can read the video frames

for video in sorted(videos_folder.glob("*.avi")):
    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print("CANNOT OPEN:", video.name)
        cap.release()
        continue

    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_read = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frames_read += 1

    cap.release()

    if frames_read == 0 or frames_read != expected_frames:
        print(
            "CHECK:", video.name,
            "| Read:", frames_read,
            "| Expected:", expected_frames
        )
    else:
        print("READ OK:", video.name, "| Frames:", frames_read)


#7# Check whether each annotation file has one row per video frame
for video in sorted(videos_folder.glob("*.avi")):
    label = labels_folder / (video.stem + ".txt")

    if not label.is_file():
        continue

    cap = cv2.VideoCapture(str(video))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    with open(label, "r", encoding="utf-8-sig") as file:
        lines = [line.strip() for line in file if line.strip()]

    # The first two lines contain the fall start and end.not always
    annotation_rows = sum(
        1 for line in lines
        if len(line.split(",")) == 6
    )

    if annotation_rows == video_frames:
        print("COUNT OK:", video.name)
    else:
        print(
            "CHECK:", video.name,
            "| Video frames:", video_frames,
            "| Annotation rows:", annotation_rows
        )
#8# check that frame numbers are in order—1, 2, 3,
##…—with no missing or repeated numbers
##Video 35 was skipped because of its previously identified annotation problem.

for label in sorted(labels_folder.glob("*.txt")):
    # Temporarily skip the file we already flagged
    if dataset.name == "Coffee_room_01" and label.name == "video (35).txt":
        continue

    with open(label, "r", encoding="utf-8-sig") as file:
        lines = [line.strip() for line in file if line.strip()]

    try:
        frame_numbers = [##This checks the frame numbers using only the six-column annotation rows. It does not change the file or verify the fall start/end values
            int(row.split(",")[0])
            for row in lines
            if len(row.split(",")) == 6
        ]

        expected = list(range(1, len(frame_numbers) + 1))

        if frame_numbers and frame_numbers == expected:
            print("ORDER OK:", label.name)
        else:
            print("CHECK FRAME ORDER:", label.name)

    except ValueError:
        print("INVALID FRAME NUMBER:", label.name)

#9# find the single-number lines wherever they appear in each file
for label in sorted(labels_folder.glob("*.txt")):
    if dataset.name == "Coffee_room_01" and label.name == "video (35).txt":
        continue

    with open(label, "r", encoding="utf-8-sig") as file:
        lines = [line.strip() for line in file if line.strip()]

    # Find lines containing just one value
    fall_values = [
        line for line in lines
        if len(line.split(",")) == 1
    ]

    print(label.name, "| Fall values:", fall_values)

#10# which codes appear in the second column
for label in sorted(labels_folder.glob("*.txt")):
    if dataset.name == "Coffee_room_01" and label.name == "video (35).txt":
        continue

    with open(label, "r", encoding="utf-8-sig") as file:
        lines = [line.strip() for line in file if line.strip()]

    codes = set()

    for line in lines:
        columns = line.split(",")

        if len(columns) == 6:
            codes.add(int(columns[1]))

    print(label.name, "| Codes:", sorted(codes))


#11# Find all Videos folders in the dataset
dataset_root = Path(r"C:\Users\haifa\Desktop\FallDataset")

for folder in sorted(dataset_root.rglob("Videos")):
    if folder.is_dir():
        annotation_folder = folder.parent / "Annotation_files" ##somw folder with s some without

        if not annotation_folder.is_dir():
            annotation_folder = folder.parent / "Annotations_files"

        print(
            "DATASET:", folder.parent.relative_to(dataset_root),
            "| Labels folder found:", annotation_folder.is_dir()
        )


#12# Find every folder containing AVI videos
video_locations = sorted({
    video.parent
    for video in dataset_root.rglob("*.avi")
    if video.is_file()
})

for folder in video_locations:
    count = len(list(folder.glob("*.avi")))

    print(
        "VIDEO FOLDER:", folder.relative_to(dataset_root),
        "| Videos:", count
    )

#13# Find every folder containing TXT files
label_locations = sorted({
    file.parent
    for file in dataset_root.rglob("*.txt")
    if file.is_file()
})

for folder in label_locations:
    files = sorted(folder.glob("*.txt"))

    print(
        "TXT FOLDER:", folder.relative_to(dataset_root),
        "| Files:", len(files),
        "| Example:", files[0].name
    )


#14# Look for files other than AVI videos in these two datasets
for name in ["Office", "Lecture_room"]:
    folder = dataset_root / name

    other_files = sorted(
        file for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() != ".avi"
    )

    print("CHECK FOLDER:", name)

    if not other_files:
        print("  Only AVI videos found.")

    for file in other_files:
        print("  OTHER FILE:", file.relative_to(dataset_root))


#15# Check whether removing exact duplicate rows would fix the annotations
for label in sorted(labels_folder.glob("*.txt")):
    video = videos_folder / (label.stem + ".avi")
    if not video.is_file():
        continue

    with open(label, "r", encoding="utf-8-sig") as file:
        rows = [
            tuple(value.strip() for value in line.strip().split(","))
            for line in file
            if len(line.strip().split(",")) == 6
        ]

    # Keep one copy of each complete row, preserving its order
    unique_rows = list(dict.fromkeys(rows))
    duplicates = len(rows) - len(unique_rows)

    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_numbers = [int(row[0]) for row in unique_rows]
    expected = list(range(1, frame_count + 1))

    if duplicates or frame_numbers != expected:
        result = (
            "EXACT DUPLICATES ONLY"
            if frame_numbers == expected
            else "NEEDS MORE REVIEW"
        )
        print(label.name, "| Extra copies:", duplicates, "|", result)


#16# Save cleaned Home_01 annotations in a separate folder
if dataset.name != "Home_02":
    raise ValueError("This step is intended for Home_02.")

clean_folder = dataset / "Annotation_files_cleaned"
clean_folder.mkdir(exist_ok=True)

for label in sorted(labels_folder.glob("*.txt")):
    with open(label, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    seen = set()
    cleaned = []
    frame_numbers = []
    removed = 0

    for line in lines:
        columns = tuple(value.strip() for value in line.strip().split(","))

        if len(columns) == 6:
            if columns in seen:
                removed += 1
                continue

            seen.add(columns)
            frame_numbers.append(int(columns[0]))

        cleaned.append(line)

    video = videos_folder / (label.stem + ".avi")
    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0 or frame_numbers != list(range(1, frame_count + 1)):
        print("NOT SAVED — REVIEW:", label.name)
        continue

    output = clean_folder / label.name
    with open(output, "w", encoding="utf-8-sig") as file:
        file.writelines(cleaned)

    print("SAVED:", label.name, "| Duplicate rows removed:", removed)