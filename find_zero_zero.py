import pandas as pd

df = pd.read_csv("dataset_index.csv")

zero_zero = df[
    (df["fall_start"] == 0) &
    (df["fall_end"] == 0)
]

print(zero_zero)
print("\nTotal 0,0 videos:", len(zero_zero))