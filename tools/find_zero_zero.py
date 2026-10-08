import pandas as pd
from falldet.config import INDEX_FILE

df = pd.read_csv(INDEX_FILE)

zero_zero = df[
    (df["fall_start"] == 0) &
    (df["fall_end"] == 0)
]

print(zero_zero)
print("\nTotal 0,0 videos:", len(zero_zero))