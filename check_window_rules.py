"""Compare window labeling rules: how many Fall windows does each rule create?"""
import csv
from config import INDEX_FILE

WINDOW, STEP = 30, 5

with open(INDEX_FILE, encoding="utf-8-sig", newline="") as f:
    falls = [r for r in csv.DictReader(f) if int(r["fall_start"]) > 0]

rules = {
    "A: 50% of window":       lambda ov, L: ov >= 0.5 * WINDOW,
    "B: 50% of fall":         lambda ov, L: ov >= 0.5 * L,
    "C: 50% of shorter one":  lambda ov, L: ov >= 0.5 * min(WINDOW, L),
}

lengths = [int(r["fall_end"]) - int(r["fall_start"]) + 1 for r in falls]
print(f"{len(falls)} falls, length min {min(lengths)}, max {max(lengths)}, "
      f"median {sorted(lengths)[len(lengths)//2]} frames\n")

for name, rule in rules.items():
    fall_windows, falls_missed = 0, 0
    for r in falls:
        s, e, n = int(r["fall_start"]), int(r["fall_end"]), int(r["frame_count"])
        L = e - s + 1
        hits = 0
        for a in range(1, n - WINDOW + 2, STEP):          # window = frames a .. a+WINDOW-1
            b = a + WINDOW - 1
            overlap = max(0, min(b, e) - max(a, s) + 1)
            if rule(overlap, L):
                hits += 1
        fall_windows += hits
        falls_missed += (hits == 0)
    print(f"{name:24} Fall windows: {fall_windows:4}   falls with NO Fall window: {falls_missed}")