import pandas as pd

csv_path = "dataset_index.csv"

df = pd.read_csv(csv_path)

mask = (
    (df["fall_start"] == 0) &
    (df["fall_end"] == 0)
)

df.loc[mask, "visual_review"] = "full_video_no_fall"

df.to_csv(csv_path, index=False)

print("Updated rows:", mask.sum())
print("Saved:", csv_path)