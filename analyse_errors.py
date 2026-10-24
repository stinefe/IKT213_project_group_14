"""Find incorrectly classified validation windows."""
import csv

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from falldet.config import OUTPUTS_DIR, PROJECT_ROOT, SPLITS_FILE
from falldet.features import FEATURE_NAMES, window_features


def main():
    # Load the saved model.
    saved = joblib.load(PROJECT_ROOT / "models" / "fall_rf.joblib")
    model = saved["model"]

    if list(saved["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError("Model feature names do not match features.py")

    # Read the video split.
    with open(SPLITS_FILE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    splits = {row["video_id"]: row["split"] for row in rows}
    if len(splits) != len(rows):
        raise ValueError("Duplicate video IDs in splits.csv")

    with np.load(OUTPUTS_DIR / "windows.npz") as data:
        video_ids = data["video_id"].astype(str)

        unknown = set(video_ids) - set(splits)
        if unknown:
            raise ValueError(f"Videos missing from splits.csv: {sorted(unknown)}")

        # Select validation windows only.
        indices = np.array([
            i for i, video in enumerate(video_ids)
            if splits[video] == "val"
        ], dtype=int)

        if len(indices) == 0:
            raise ValueError("No validation windows found")

        X = data["X"]

        if X.shape[1] != saved["window_frames"]:
            raise ValueError("Window length does not match the model")

        features = np.stack([
            window_features(X[i]) for i in indices
        ])
        true = data["y"][indices]
        predicted = model.predict(features)

        results = pd.DataFrame({
            "window_index": indices,
            "video_id": video_ids[indices],
            "window_start": data["window_start"][indices],
            "window_end": data["window_end"][indices],
            "fps": data["fps"][indices],
            "true_label": np.where(true == 1, "Fall", "Normal"),
            "predicted_label": np.where(predicted == 1, "Fall", "Normal"),
            "core_missing_fraction": data["core_missing_fraction"][indices],
        })

    # Include feature values to help investigate errors.
    for column, name in enumerate(FEATURE_NAMES):
        results[name] = features[:, column]

    errors = results.loc[true != predicted].copy()
    errors["error_type"] = np.where(
        errors["true_label"] == "Fall",
        "False Negative",
        "False Positive",
    )
    errors["review_notes"] = ""
    errors = errors.sort_values(["video_id", "window_start"])

    output = OUTPUTS_DIR / "validation_errors.csv"
    errors.to_csv(output, index=False, encoding="utf-8-sig")

    print("Validation windows:", len(indices))
    print("Confusion matrix (Normal, Fall):")
    print(confusion_matrix(true, predicted, labels=[0, 1]))
    print("\nErrors:")
    print(errors["error_type"].value_counts())
    print("Videos with errors:", errors["video_id"].nunique())
    print("\nSaved:", output)


if __name__ == "__main__":
    main()