import csv
from falldet.config import INDEX_FILE

fall_count = 0
short_fall_count = 0

with INDEX_FILE.open("r", encoding="utf-8-sig", newline="") as file:
    for row in csv.DictReader(file):
        start = int(row["fall_start"])
        end = int(row["fall_end"])

        # These videos have no annotated fall interval.
        if start == 0 and end == 0:
            continue

        # Count both the start frame and the end frame.
        fall_length = end - start + 1
        fall_count += 1

        if fall_length < 15:
            short_fall_count += 1

        print(f"{row['video_id']} | Fall length: {fall_length} frames")

print("\nTotal annotated falls:", fall_count)
print("Falls shorter than 15 frames:", short_fall_count)