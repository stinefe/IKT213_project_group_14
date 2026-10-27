"""Live fall detection on the laptop webcam (or Jetson camera).

Pipeline per frame (same steps as the training data):
  camera frame -> YOLO11n-Pose -> pick one person -> keypoints (17, 3)
  -> keep the last WINDOW frames -> window_features() -> classifier -> FALL / OK
Display only: blurred copy of the frame (faces hidden) + skeleton + label.
The sharp frame is used for analysis only and is never shown or saved.

Run from the repo root:   python live_detect.py
Press q to quit.
"""
import os
import platform
import time
from collections import deque

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

from falldet.config import MODEL_PATH            # YOLO pose weights (same as batch extraction)
from falldet.pose_preprocessing import interpolate_short_gaps  # same gap-filling as training
try:
    from falldet.features import window_features  # if features.py lives inside falldet/
except ImportError:
    from features import window_features           # if features.py is in the repo root
from falldet.face_blur import blur_faces_from_result  # privacy blur for the display copy
from alerts.alerts import AlertClient                # MQTT alerts (issue #11)

# ---- MUST MATCH TRAINING (check train_baseline.py) --------------------------
CLASSIFIER_FILE = "models/fall_rf.joblib"      # Random Forest saved by train_baseline.py
WINDOW = 30            # WINDOW_LENGTH in the window-building script
TRAIN_FPS = 25         # Le2i videos are 24-25 fps
FALL_LABEL = 1         # choose_label(): 0 = Normal, 1 = Fall
MAX_GAP_SECONDS = 0.12       # fill gaps shorter than this (same as training)
MAX_MISSING_FRACTION = 0.30  # skip windows with more missing core joints (same as training)
CORE_JOINTS = [5, 6, 11, 12]
# ---- after-the-model rules (tune these by testing) ---------------------------
UPRIGHT_ANGLE = 30     # torso closer than this to vertical = standing/sitting up
CONFIRM_SECONDS = 2.0  # must stay down this long after a "fall" to count as fallen
RECOVER_SECONDS = 3.0  # must be upright this long before the alarm clears
# ---- same values as the batch pose extraction --------------------------------
NMS_IOU = 0.5
KEYPOINT_CONF = 0.5
DETECTION_CONF = 0.1
MIN_GOOD_KEYPOINTS = 5
# ---- where it runs (set by Docker, defaults work on the laptop) --------------
HEADLESS = os.environ.get("HEADLESS") == "1"   # no window, print state changes
DEVICE = os.environ.get("DEVICE") or ("0" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------------------------------


def open_camera():
    """CAMERA env var if set (e.g. a stream URL in Docker), else laptop webcam / Jetson CSI."""
    source = os.environ.get("CAMERA")
    if source:
        return cv2.VideoCapture(int(source) if source.isdigit() else source)
    if platform.machine() == "aarch64":  # Jetson, outside Docker
        pipeline = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # 640x480 = 4:3, same shape as the training videos (features.py assumes 4:3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def pose_one_frame(model, frame, prev_center):
    """Same person-picking logic as extract_one() in the batch script, for one frame."""
    result = model.predict(frame, device=DEVICE, imgsz=640, conf=DETECTION_CONF,
                           iou=NMS_IOU, verbose=False)[0]
    points = np.full((17, 3), np.nan, dtype=np.float32)
    raw = len(result.boxes) if result.boxes is not None else 0

    valid = np.array([], dtype=int)
    if raw and result.keypoints is not None and result.keypoints.conf is not None:
        kp_conf = result.keypoints.conf.cpu().numpy()
        valid = np.where((kp_conf >= KEYPOINT_CONF).sum(axis=1) >= MIN_GOOD_KEYPOINTS)[0]

    if len(valid) >= 1:
        centers = result.boxes.xywhn.cpu().numpy()[valid, :2]
        if len(valid) == 1:
            pick = 0
        elif prev_center is None:
            pick = int(np.argmax(result.boxes.conf.cpu().numpy()[valid]))
        else:
            pick = int(np.argmin(np.linalg.norm(centers - prev_center, axis=1)))
        idx = valid[pick]
        prev_center = centers[pick]

        xy = result.keypoints.xyn[idx].cpu().numpy().copy()
        conf = result.keypoints.conf[idx].cpu().numpy()
        xy[conf < KEYPOINT_CONF] = np.nan
        points[:, :2] = xy
        points[:, 2] = conf

    return points, prev_center, result


def main():
    pose_model = YOLO(str(MODEL_PATH))
    saved = joblib.load(CLASSIFIER_FILE)
    threshold = None
    if isinstance(saved, dict):
        # train_baseline.py saved a dict: the model plus extra info
        print("Saved file contains:", {k: type(v).__name__ for k, v in saved.items()})
        classifier = next(v for v in saved.values() if hasattr(v, "predict"))
        for key in ("threshold", "decision_threshold", "best_threshold"):
            if key in saved:
                threshold = float(saved[key])
                print(f"Using saved threshold {threshold}")
    else:
        classifier = saved
    cap = open_camera()
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    alerts = AlertClient()
    alerts.start()
    fallen_since = None   # when the current fall was confirmed

    window = deque(maxlen=WINDOW)  # last WINDOW frames of keypoints
    prev_center = None
    last_t = time.time()

    # State machine on top of the classifier:
    #   NORMAL --model says fall--> CHECKING --still down after CONFIRM_SECONDS--> FALLEN
    #   CHECKING --upright again--> NORMAL   (it was just bending down)
    #   FALLEN --upright for RECOVER_SECONDS--> NORMAL   (stays FALLEN while on the floor)
    state = "NORMAL"
    check_start = None
    upright_since = None
    torso_angle = None
    last_printed = None
    print(f"Running on device={DEVICE}, headless={HEADLESS}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No camera frame")
            break

        points, prev_center, result = pose_one_frame(pose_model, frame, prev_center)
        window.append(points)

        now = time.time()
        model_says_fall = False
        torso_angle = None   # None = can't see the body well enough to judge

        if len(window) == WINDOW:
            # Same cleaning as training: fill short gaps, then quality check
            processed, _, remaining = interpolate_short_gaps(
                np.stack(window).copy(), TRAIN_FPS, max_gap_seconds=MAX_GAP_SECONDS)
            missing = float(remaining[:, CORE_JOINTS].mean())

            if missing <= MAX_MISSING_FRACTION:   # training skipped worse windows
                processed[:, :, :2][remaining] = np.nan
                feats = window_features(processed)
                torso_angle = float(feats[2])     # torso_angle_end: 0 = upright, 90 = lying
                if np.isnan(torso_angle):
                    torso_angle = None
                x = feats.reshape(1, -1)
                if threshold is not None:
                    fall_col = list(classifier.classes_).index(FALL_LABEL)
                    model_says_fall = classifier.predict_proba(x)[0, fall_col] >= threshold
                else:
                    model_says_fall = classifier.predict(x)[0] == FALL_LABEL

        upright = torso_angle is not None and torso_angle < UPRIGHT_ANGLE

        # ---- state machine ----
        if state == "NORMAL":
            if model_says_fall:
                state, check_start = "CHECKING", now
        elif state == "CHECKING":
            if upright:
                state = "NORMAL"                  # came back up -> bending, not a fall
            elif now - check_start >= CONFIRM_SECONDS:
                state, upright_since = "FALLEN", None
                fallen_since = now
                alerts.fall(torso_angle=torso_angle)      # once per fall
        elif state == "FALLEN":
            if upright:
                upright_since = upright_since or now
                if now - upright_since >= RECOVER_SECONDS:
                    state = "NORMAL"              # standing again for a while
                    alerts.recovered(upright_since - fallen_since)
            else:
                upright_since = None              # still down -> keep alarm on

        if len(window) < WINDOW:
            label, color = "warming up...", (200, 200, 200)
        elif state == "NORMAL":
            label, color = "OK", (0, 200, 0)
        elif state == "CHECKING":
            label, color = f"checking... {now - check_start:.1f}s", (0, 200, 255)
        else:
            label, color = "FALL!", (0, 0, 255)

        now = time.time()
        fps = 1.0 / max(now - last_t, 1e-6)
        last_t = now
        alerts.heartbeat(state, fps)   # "I'm alive" every 10 s

        if HEADLESS:   # no screen (Docker / room corner): print when the state changes
            if state != last_printed:
                angle_txt = "?" if torso_angle is None else f"{torso_angle:.0f}"
                print(f"{time.strftime('%H:%M:%S')} {state:8s} fps={fps:.1f} torso={angle_txt}")
                last_printed = state
            continue

        # Privacy: YOLO already ran on the SHARP frame (above). Only the copy we show
        # is blurred, then the skeleton is drawn on top of the blurred copy.
        # (result.plot() alone would draw on the sharp original, so img= is required.)
        shown = blur_faces_from_result(frame, result)
        shown = result.plot(img=shown)
        cv2.putText(shown, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        angle_txt = "?" if torso_angle is None else f"{torso_angle:.0f}"
        cv2.putText(shown, f"{fps:.1f} fps (trained on {TRAIN_FPS}) | torso {angle_txt} deg",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("fall detection", shown)
        if cv2.waitKey(1) == ord("q"):
            break

    alerts.stop()
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()