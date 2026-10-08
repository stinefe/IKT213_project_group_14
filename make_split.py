import pandas as pd
from sklearn.model_selection import train_test_split

# File path to dataset index CSV file
INDEX_CSV = "dataset_index.csv"

# File path to output CSV file for train/test split
OUTPUT_CSV = "splits.csv"

# Specify the room to be used for the test set
TEST_ROOM = "Home_02"

# Fraction of the training set to be used for validation
VAL_FRAC= 0.2  

SEED = 42  # Random seed for reproducibility

df = pd.read_csv(INDEX_CSV)
print(f"Total records in dataset: {len(df)}")

label_map = {
    "fall_interval": "Fall",
    "zero_zero": "Normal",
}
df["label"] = df["annotation_status"].map(label_map)
assert df["label"].notnull().all(), "Some records have unknown annotation_status"

is_test = df["dataset"] == TEST_ROOM
test_df = df[is_test].copy()
rest_df = df[~is_test].copy()

test_df["split"] = "test"

print("Test:", len(test_df), "| Resten:", len(rest_df))
print(test_df["label"].value_counts())

train_df, val_df = train_test_split(
    rest_df,
    test_size=VAL_FRAC,
    stratify=rest_df["label"],
    random_state=SEED
)

train_df["split"] = "train"
val_df["split"] = "val"

df_split = pd.concat([train_df, val_df, test_df])
print(len(df_split), "records after split")
print(pd.crosstab(df_split["split"], df_split["label"], margins=True))

# Sanity checks
assert len(df_split) == len(df), "Videoer har forsvunnet eller blitt duplisert"
assert df_split["video_id"].is_unique, "En video ligger i mer enn én split"
assert (df_split.loc[df_split["dataset"] == TEST_ROOM, "split"] == "test").all()
assert (df_split.loc[df_split["dataset"] != TEST_ROOM, "split"] != "test").all()

# Lagre
df_split[["video_id", "dataset", "label", "split"]].to_csv(OUTPUT_CSV, index=False)
print("Lagret til", OUTPUT_CSV)