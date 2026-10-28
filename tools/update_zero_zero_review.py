import pandas as pd
from falldet.config import INDEX_FILE

df = pd.read_csv(INDEX_FILE)

mask = (
    (df["fall_start"] == 0) &
    (df["fall_end"] == 0)
)

df.loc[mask, "visual_review"] = "full_video_no_fall"

df.to_csv(INDEX_FILE, index=False)

print("Updated rows:", mask.sum())
print("Saved:", INDEX_FILE)