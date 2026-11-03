"""Final evaluation on the TEST set (the Office room, never seen during training).
"""
import csv

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from falldet.config import OUTPUTS_DIR, PROJECT_ROOT, SPLITS_FILE
from falldet.features import FEATURE_NAMES, window_features

WINDOWS_FILE = OUTPUTS_DIR / "windows.npz"
MODEL_FILE = PROJECT_ROOT / "models" / "fall_rf.joblib"


def main():
    # 1. The saved model, with the threshold chosen on validation
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    threshold = float(saved.get("threshold", 0.5))
    if saved.get("feature_names") != FEATURE_NAMES:
        raise SystemExit("features.py has changed since this model was trained: retrain first.")
    print(f"Model: {MODEL_FILE.name}, threshold {threshold}, "
          f"scikit-learn {saved.get('sklearn_version')}")

    # 2. Test windows only
    data = np.load(WINDOWS_FILE)
    X, y, video_ids = data["X"], data["y"], data["video_id"]
    with open(SPLITS_FILE, encoding="utf-8", newline="") as f:
        splits = {r["video_id"]: r["split"] for r in csv.DictReader(f)}
    test = np.array([splits[v] == "test" for v in video_ids])
    if not test.any():
        raise SystemExit("No test windows found: check data/splits.csv.")

    F = np.stack([window_features(w) for w in X[test]])
    yt, ids = y[test], video_ids[test]
    rooms = sorted({v.split("/")[0] for v in ids})
    print(f"Test windows: {len(yt)} from {len(set(ids))} videos ({', '.join(rooms)}), "
          f"Fall share {yt.mean():.1%}")

    # 3. Window level, at the chosen threshold
    proba = model.predict_proba(F)[:, list(model.classes_).index(1)]
    pred = (proba >= threshold).astype(int)
    print(f"\nConfusion matrix at threshold {threshold} (rows = true Normal/Fall, cols = predicted):")
    print(confusion_matrix(yt, pred, labels=[0, 1]))
    print(classification_report(yt, pred, labels=[0, 1],
                                target_names=["Normal", "Fall"], zero_division=0))

    # 4. Event level: is each fall caught, does each video get a false alarm?
    fall_vids = sorted(set(ids[yt == 1]))
    all_vids = sorted(set(ids))
    missed = [v for v in fall_vids if not pred[(ids == v) & (yt == 1)].any()]
    false_alarm = [v for v in all_vids if pred[(ids == v) & (yt == 0)].any()]
    no_fall_alarms = [v for v in false_alarm if v not in fall_vids]

    print(f"Falls detected: {len(fall_vids) - len(missed)}/{len(fall_vids)}")
    for v in missed:
        print("  MISS", v)
    print(f"Videos with a false alarm window: {len(false_alarm)}/{len(all_vids)}")
    print(f"  in no-fall videos: {len(no_fall_alarms)} | "
          f"in fall videos (before the fall): {len(false_alarm) - len(no_fall_alarms)}")
    for v in no_fall_alarms:
        print("  FALSE ALARM", v)


if __name__ == "__main__":
    main()