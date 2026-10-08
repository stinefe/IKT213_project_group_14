"""Baseline fall classifier. Plugs into #6 (windows.npz) and #7 (splits.csv)."""
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from falldet.config import OUTPUTS_DIR
from falldet.features import FEATURE_NAMES, window_features

USE_DUMMY_DATA = True                          # switch to False when #6 and #7 are done
WINDOWS_FILE = OUTPUTS_DIR / "windows.npz"     # from #6: X (n,30,17,3), y (n,), video_id (n,)
SPLITS_FILE = OUTPUTS_DIR / "splits.csv"       # from #7: video_id,split


def load_dummy(n=600, n_videos=40, seed=0):
    """Fake data with the agreed shapes, only to test that the code runs."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.3, 0.7, (n, 30, 17, 3)).astype(np.float32)
    y = (rng.random(n) < 0.1).astype(int)       # ~10% Fall, like the real data
    X[y == 1, 15:, :, 1] += 0.25                # fake fall: body moves down in the 2nd half
    video_ids = np.array([f"dummy/video ({i % n_videos})" for i in range(n)])
    splits = {f"dummy/video ({i})": "train" if i < 28 else "val" if i < 34 else "test"
              for i in range(n_videos)}
    return X, y, video_ids, splits


def load_real():
    data = np.load(WINDOWS_FILE)
    with open(SPLITS_FILE, encoding="utf-8", newline="") as f:
        splits = {r["video_id"]: r["split"] for r in csv.DictReader(f)}
    return data["X"], data["y"], data["video_id"], splits


def main():
    X, y, video_ids, splits = load_dummy() if USE_DUMMY_DATA else load_real()

    # 1. Each window -> feature vector
    F = np.stack([window_features(w) for w in X])
    split = np.array([splits[v] for v in video_ids])
    train, val = split == "train", split == "val"      # test set: only used once, at the very end
    print(f"Windows: train {train.sum()}, val {val.sum()}, Fall share {y.mean():.1%}")

    # 2. Model: fill missing values, then a Random Forest that weights the rare Fall class up
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0),
    )
    model.fit(F[train], y[train])

    # 3. Evaluate on validation
    pred = model.predict(F[val])
    print("\nConfusion matrix (rows = true Normal/Fall, cols = predicted):")
    print(confusion_matrix(y[val], pred, labels=[0, 1]))
    print(classification_report(y[val], pred, labels=[0, 1],
                                target_names=["Normal", "Fall"], zero_division=0))

    # 4. Which features mattered most?
    importances = model[-1].feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda t: -t[1]):
        print(f"  {name:20} {imp:.2f}")


if __name__ == "__main__":
    main()