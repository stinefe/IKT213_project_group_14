import csv
import numpy as np
from config import OUTPUTS_DIR

import sys
RUN = sys.argv[1] if len(sys.argv) > 1 else "" # when running in terminal add the file number name
POSES_DIR = OUTPUTS_DIR / (f"poses_{RUN}" if RUN else "poses")
SUMMARY_FILE = OUTPUTS_DIR / (f"pose_summary_{RUN}.csv" if RUN else "pose_summary.csv")
print("Checking:", POSES_DIR.name)

with open(SUMMARY_FILE, encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["status"] in ("ok", "cached")]

frames = np.array([int(r["frames"]) for r in rows])
multi = np.array([int(r["multi_person"]) for r in rows])
none = np.array([int(r["no_person"]) for r in rows])
core = np.array([float(r["core_joints_ok_pct"]) for r in rows])
fall = np.array([float(r["fall_core_ok_pct"]) for r in rows if r["fall_core_ok_pct"]])

print(f"Videos: {len(rows)}   Total frames: {frames.sum()}")
print(f"No person:    {none.sum()} frames ({100*none.sum()/frames.sum():.1f}%)")
print(f"Multi person: {multi.sum()} frames ({100*multi.sum()/frames.sum():.1f}%)")
print(f"Core joints OK, average:        {core.mean():.1f}%")
print(f"Core joints OK during falls:    {fall.mean():.1f}% (over {len(fall)} fall videos)")

print("\nWorst 5 videos (core joints OK %):")
for r in sorted(rows, key=lambda r: float(r["core_joints_ok_pct"]))[:5]:
    print(f"  {r['video_id']:28} {r['core_joints_ok_pct']:>5}%  multi={r['multi_person']}  none={r['no_person']}")

print("\nMost multi-person frames:")
for r in sorted(rows, key=lambda r: -int(r["multi_person"]))[:5]:
    print(f"  {r['video_id']:28} {r['multi_person']} of {r['frames']} frames")

    from pathlib import Path
CORE = [5, 6, 11, 12]
none_f = multi_f = lowconf_f = total_f = 0
for npz in (POSES_DIR).glob("*.npz"):
    d = np.load(npz)
    s, e = int(d["fall_start"]), int(d["fall_end"])
    if s == 0:
        continue
    kp, pc = d["keypoints"][s-1:e], d["person_counts"][s-1:e]
    bad = np.isnan(kp[:, CORE, 0]).any(axis=1)
    total_f += len(kp)
    none_f += int(np.sum(bad & (pc == 0)))
    multi_f += int(np.sum(bad & (pc > 1)))
    lowconf_f += int(np.sum(bad & (pc == 1)))
print(f"\nFall frames: {total_f}, missing core joints because of:")
print(f"  no person:      {none_f}")
print(f"  multi person:   {multi_f}")
print(f"  low confidence: {lowconf_f}")