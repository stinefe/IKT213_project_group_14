"""Save one frame with YOLO's detections drawn, and list the bad frames of a video."""
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from config import INDEX_FILE, OUTPUTS_DIR, MODEL_PATH, resolve_dataset_path

VIDEO_ID = "Home_02/video (58)"
FRAME = 150              # 1-based frame number
DETECTION_CONF = 0.25    # try 0.1 to see if more people are found

# 1. Find the video in the index
with open(INDEX_FILE, encoding="utf-8-sig", newline="") as f:
    record = next(r for r in csv.DictReader(f) if r["video_id"] == VIDEO_ID)
video = resolve_dataset_path(record["video_path"])

# 2. List which frames were bad in the batch run
npz = OUTPUTS_DIR / "poses" / (VIDEO_ID.replace("/", "__").replace(" (", "_").replace(")", "") + ".npz")
if npz.exists():
    counts = np.load(npz)["person_counts"]
    print("No-person frames:   ", (np.where(counts == 0)[0] + 1).tolist())
    print("Multi-person frames:", (np.where(counts > 1)[0] + 1).tolist())
    print(f"Fall interval: {record['fall_start']}-{record['fall_end']}")

# 3. Jump to the frame and run YOLO on it
cap = cv2.VideoCapture(str(video))
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME - 1)
ok, image = cap.read()
cap.release()
if not ok:
    raise ValueError(f"Could not read frame {FRAME}")

result = YOLO(str(MODEL_PATH)).predict(image, conf=DETECTION_CONF, iou=0.5,
                                        imgsz=640, device="cpu", verbose=False)[0]

# 4. Print what YOLO found and save the picture
for i, c in enumerate(result.boxes.conf.cpu().numpy() if result.boxes is not None else []):
    print(f"Person {i + 1}: confidence {c:.2f}")
out = OUTPUTS_DIR / f"frame_{VIDEO_ID.replace('/', '_').replace(' ', '')}_{FRAME}.jpg"
cv2.imwrite(str(out), result.plot())
print("Saved:", out)