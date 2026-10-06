"""Batch pose extraction for all indexed videos (issue #4).

Runs YOLO11n-Pose on every video in dataset_index.csv, saves one .npz
per video, and writes a summary CSV with quality numbers per video.
Same settings and missing-keypoint rules as extract_pose.py.
"""
import csv
import re
import time

import numpy as np
from ultralytics import YOLO

from config import (INDEX_FILE, OUTPUTS_DIR, MODEL_PATH,
                    require_dataset_root, resolve_dataset_path)

KEYPOINT_CONF = 0.5          # same as extract_pose.py
DETECTION_CONF = 0.25        # same as extract_pose.py
CORE_JOINTS = [5, 6, 11, 12] # left/right shoulder, left/right hip (COCO order)
LIMIT = 3                    # testing on 3 videos first; will set to None for all 129 if it works

POSES_DIR = OUTPUTS_DIR / "poses"
SUMMARY_FILE = OUTPUTS_DIR / "pose_summary.csv"


def safe_name(video_id):
    """'Home_01/video (1)' -> 'Home_01__video_1' (safe as a filename)."""
    name = video_id.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def extract_one(model, video):
    """Run pose on one video. Returns keypoints (N,17,3) and person counts (N,)."""
    results = model.predict(
        source=str(video), stream=True, device="cpu", imgsz=640,
        conf=DETECTION_CONF, vid_stride=1, save=False, verbose=False,
    )
    all_keypoints, person_counts = [], []

    for result in results:
        count = len(result.boxes) if result.boxes is not None else 0
        points = np.full((17, 3), np.nan, dtype=np.float32)

        if count == 1 and result.keypoints is not None and result.keypoints.conf is not None:
            xy = result.keypoints.xyn[0].cpu().numpy().copy()
            conf = result.keypoints.conf[0].cpu().numpy()
            xy[conf < KEYPOINT_CONF] = np.nan
            points[:, :2] = xy
            points[:, 2] = conf

        all_keypoints.append(points)
        person_counts.append(count)

    return np.stack(all_keypoints), np.asarray(person_counts)


def summarize(record, keypoints, counts, status, seconds):
    """One row of quality numbers for the summary CSV."""
    n = len(keypoints)
    core_ok = ~np.isnan(keypoints[:, CORE_JOINTS, 0]).any(axis=1)
    row = {
        "video_id": record["video_id"],
        "status": status,
        "frames": n,
        "one_person": int(np.sum(counts == 1)),
        "no_person": int(np.sum(counts == 0)),
        "multi_person": int(np.sum(counts > 1)),
        "core_joints_ok_pct": round(100 * core_ok.mean(), 1) if n else 0,
        "fall_core_ok_pct": "",
        "seconds": round(seconds, 1),
    }
    start, end = int(record["fall_start"]), int(record["fall_end"])
    if start > 0:  # frames are 1-based and the interval is inclusive
        row["fall_core_ok_pct"] = round(100 * core_ok[start - 1:end].mean(), 1)
    return row


def main():
    require_dataset_root()
    POSES_DIR.mkdir(parents=True, exist_ok=True)

    with open(INDEX_FILE, "r", encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    if LIMIT:
        records = records[:LIMIT]
    total = len(records)
    print(f"Videos to process: {total}")

    model = YOLO(str(MODEL_PATH))  # load once, reuse for every video
    summary = []

    for i, record in enumerate(records, start=1):
        video_id = record["video_id"]
        out_file = POSES_DIR / f"{safe_name(video_id)}.npz"
        started = time.time()

        if out_file.exists():  # resume: already done in an earlier run
            data = np.load(out_file)
            keypoints, counts = data["keypoints"], data["person_counts"]
            summary.append(summarize(record, keypoints, counts, "cached", 0))
            print(f"[{i}/{total}] skip (already done): {video_id}")
            continue

        try:
            video = resolve_dataset_path(record["video_path"])
            if not video.is_file():
                raise FileNotFoundError(video)

            keypoints, counts = extract_one(model, video)

            expected = int(record["frame_count"])
            if len(keypoints) != expected:
                raise ValueError(f"processed {len(keypoints)} frames, expected {expected}")

            np.savez_compressed(
                out_file,
                video_id=video_id,
                dataset=record["dataset"],
                frame_numbers=np.arange(1, len(keypoints) + 1),
                keypoints=keypoints,
                person_counts=counts,
                fall_start=int(record["fall_start"]),
                fall_end=int(record["fall_end"]),
                annotation_status=record["annotation_status"],
                keypoint_confidence_threshold=KEYPOINT_CONF,
            )
            seconds = time.time() - started
            summary.append(summarize(record, keypoints, counts, "ok", seconds))
            print(f"[{i}/{total}] {video_id}: {len(keypoints)} frames, {seconds:.0f}s")

        except Exception as error:  # log it and keep going with the next video
            print(f"[{i}/{total}] FAILED {video_id}: {error}")
            summary.append({"video_id": video_id, "status": f"error: {error}"})

    fields = ["video_id", "status", "frames", "one_person", "no_person",
              "multi_person", "core_joints_ok_pct", "fall_core_ok_pct", "seconds"]
    with open(SUMMARY_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    failed = [r for r in summary if str(r["status"]).startswith("error")]
    print(f"\nDone. {total - len(failed)} ok, {len(failed)} failed.")
    print("Pose files:", POSES_DIR)
    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()