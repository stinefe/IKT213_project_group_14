"""Testskript: YOLO-pose på Le2i-videoer.

Kjør fra prosjektroten:
    uv add ultralytics
    uv run scripts/yolo_pose_test.py --input "scripts/Home_01/Home_01/Videos/video (1).avi"

Taster: q = avslutt, mellomrom = pause.
Modellen (yolo11n-pose.pt, ~6 MB) lastes ned automatisk første gang.
"""
import argparse
import time

import cv2 as cv
import numpy as np
from ultralytics import YOLO

# COCO-17 (rekkefølgen YOLO-pose bruker)
KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear",
            "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
            "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Sti til video")
parser.add_argument("--model", default="yolo11n-pose.pt")
parser.add_argument("--kp-thr", type=float, default=0.3, help="Min. confidence per keypoint")
parser.add_argument("--imgsz", type=int, default=640, help="Inputstørrelse til modellen")
parser.add_argument("--blank", action="store_true", help="Tegn skjelett på svart bakgrunn (personvern)")
parser.add_argument("--scale", type=float, default=2.0, help="Forstørr visningen")
args = parser.parse_args()

model = YOLO(args.model)
cap = cv.VideoCapture(args.input)
if not cap.isOpened():
    raise SystemExit(f"Kunne ikke åpne {args.input}")

fps = cap.get(cv.CAP_PROP_FPS) or 25
delay = int(1000 / fps)
paused = False

while True:
    if not paused:
        ok, frame = cap.read()
        if not ok:
            break
        idx = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        t0 = time.perf_counter()
        res = model(frame, imgsz=args.imgsz, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        canvas = np.zeros_like(frame) if args.blank else frame.copy()
        n_people = 0

        if res.keypoints is not None and len(res.keypoints) > 0:
            # Ta personen med høyest boks-confidence (én person per video i Le2i)
            best = int(res.boxes.conf.argmax())
            xy = res.keypoints.xy[best].cpu().numpy()      # (17, 2)
            conf = res.keypoints.conf[best].cpu().numpy()  # (17,)
            n_people = len(res.keypoints)

            x1, y1, x2, y2 = res.boxes.xyxy[best].cpu().numpy().astype(int)
            cv.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), 1)

            for a, b in SKELETON:
                if conf[a] > args.kp_thr and conf[b] > args.kp_thr:
                    cv.line(canvas, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0, 255, 0), 2)
            for i, (x, y) in enumerate(xy):
                if conf[i] > args.kp_thr:
                    cv.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1)

            n_ok = int((conf > args.kp_thr).sum())
            info = f"frame {idx}  {ms:.0f} ms  kp {n_ok}/17  pers {n_people}"
        else:
            info = f"frame {idx}  {ms:.0f} ms  INGEN PERSON"

        cv.putText(canvas, info, (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        show = cv.resize(canvas, None, fx=args.scale, fy=args.scale)

    cv.imshow("YOLO-pose test", show)
    key = cv.waitKey(delay) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        paused = not paused

cap.release()
cv.destroyAllWindows()