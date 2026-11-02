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
import threading
import time
import warnings
from collections import deque

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

from falldet.config import MODEL_PATH            # YOLO pose weights (same as batch extraction)
from falldet.pose_preprocessing import interpolate_short_gaps  # same gap-filling as training
from falldet.features import FEATURE_NAMES, window_features
TORSO_IDX = FEATURE_NAMES.index("torso_angle_end")  # by name, so adding features can't break it
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
CONFIRM_SECONDS = 2.0  # must be LYING this long (in a row) after a "fall" to count as fallen
RECOVER_SECONDS = 3.0  # must be upright this long before the alarm clears
# Legs = line from mid-hip to mid-ankle. 0 deg = legs vertical (on your feet), 90 = flat on the floor.
ON_FEET_ANGLE = 40     # legs steeper than this: standing / bending / squatting -> not a fall
LYING_ANGLE = 60       # legs flatter than this: lying or sitting on the floor
GRACE_SECONDS = 1.0    # after "fall?", give the body time to land before judging the legs
MAX_CHECK_SECONDS = 8  # never stay yellow longer than this without deciding
# ---- same values as the batch pose extraction --------------------------------
NMS_IOU = 0.5
KEYPOINT_CONF = 0.5
DETECTION_CONF = 0.1
MIN_GOOD_KEYPOINTS = 5
# ---- where it runs (set by Docker, defaults work on the laptop) --------------
HEADLESS = os.environ.get("HEADLESS") == "1"   # no window, print state changes
DEVICE = os.environ.get("DEVICE") or ("0" if torch.cuda.is_available() else "cpu")
TIMING = os.environ.get("TIMING") == "1"       # print ms per step (for finding slow parts)
# ------------------------------------------------------------------------------


class CameraThread:
    """Reads frames in the background so the main loop never waits for the camera.

    Without this, camera (~100 ms) and YOLO (~90 ms) run one after the other.
    With it, they run at the same time, so fps is set by the slowest step, not the sum.
    """

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.frame_id = 0          # goes up by 1 for every new frame
        self.read_fps = 0.0        # how fast the CAMERA side delivers (for lag diagnosis)
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        count, since = 0, time.time()
        while self.running:
            ok, frame = self.cap.read()
            count += 1
            if time.time() - since >= 2.0:
                self.read_fps = count / (time.time() - since)
                count, since = 0, time.time()
            if not ok:
                self.running = False
                break
            with self.lock:        # lock = "don't read while I'm writing"
                self.frame = frame
                self.frame_time = time.time()   # when the picture was TAKEN
                self.frame_id += 1

    def read(self, last_id):
        """Wait for a frame NEWER than last_id, so the same frame is never processed twice.

        Returns (ok, frame, frame_id, capture_time).
        """
        while self.running:
            with self.lock:
                if self.frame_id > last_id:
                    return True, self.frame.copy(), self.frame_id, self.frame_time
            time.sleep(0.002)
        return False, None, last_id, None

    def release(self):
        self.running = False
        time.sleep(0.05)           # let the background read finish
        self.cap.release()


WINDOW_SECONDS = (WINDOW - 1) / TRAIN_FPS   # 30 frames at 25 fps = 1.16 s from first to last


def resample_to_train_fps(history):
    """Turn real frames (any fps) into exactly WINDOW frames at TRAIN_FPS.

    history: list of (capture_time, points(17, 3)), oldest first.
    Returns a (WINDOW, 17, 3) array covering the last WINDOW_SECONDS, or None if
    the history does not reach back that far yet.

    Like drawing 30 evenly spaced dots on a line through the real frames: each
    new dot lies between the two real frames around it (linear interpolation).
    A joint missing in either neighbour stays missing (NaN), so the gap filling
    and the quality check afterwards behave exactly like in training.
    """
    times = np.array([t for t, _ in history])
    points = np.stack([p for _, p in history])
    end = times[-1]
    if len(times) < 2 or times[0] > end - WINDOW_SECONDS:
        return None

    targets = np.linspace(end - WINDOW_SECONDS, end, WINDOW)
    before = np.clip(np.searchsorted(times, targets, side="right") - 1, 0, len(times) - 2)
    after = before + 1
    w = (targets - times[before]) / np.maximum(times[after] - times[before], 1e-6)
    w = np.clip(w, 0.0, 1.0)[:, None, None]

    a, b = points[before], points[after]
    mixed = a * (1 - w) + b * w
    # exactly on a real frame: copy it (avoids NaN * 0 = NaN from the other neighbour)
    return np.where(w <= 0, a, np.where(w >= 1, b, mixed)).astype(np.float32)


L_HIP, R_HIP, L_ANKLE, R_ANKLE = 11, 12, 15, 16
IMG_ASPECT = 4 / 3   # same correction as features.py: x and y are normalized separately


def legs_angle(window, k=5):
    """Angle of the hip->ankle line from vertical over the last k frames, or None.

    Bending down to pick something up: the torso tips over, but the legs stay
    roughly vertical (you are still on your feet). After a fall the legs lie on
    the floor too. One visible ankle is enough.
    """
    w = window[-k:]
    hip = (w[:, L_HIP, :2] + w[:, R_HIP, :2]) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # both ankles missing -> NaN
        ankle = np.nanmean(w[:, [L_ANKLE, R_ANKLE], :2], axis=1)   # one visible ankle is enough
    legs = ankle - hip
    angle = np.degrees(np.arctan2(np.abs(legs[:, 0] * IMG_ASPECT), np.abs(legs[:, 1])))
    angle = angle[~np.isnan(angle)]
    return float(angle.mean()) if len(angle) else None


def open_camera():
    """CAMERA env var if set (e.g. a stream URL in Docker), else laptop webcam / Jetson CSI."""
    source = os.environ.get("CAMERA")
    if source:
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        # Network stream (Docker on the Jetson): tell FFmpeg to keep NO backlog.
        # Default FFmpeg buffers seconds of video "to be safe" -> the picture lags behind.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                              "fflags;nobuffer|flags;low_delay|framedrop;1")
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    if platform.machine() == "aarch64":  # Jetson, outside Docker
        pipeline = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # MJPG: many webcams deliver far more fps in this format than the default
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
    cam = CameraThread(cap)
    frame_id = 0

    alerts = AlertClient()
    alerts.start()
    fallen_since = None   # when the current fall was confirmed

    # Real frames with their capture time. 200 is plenty for 1.16 s at any webcam fps;
    # resample_to_train_fps() turns them into the 30 frames at 25 fps the model expects.
    history = deque(maxlen=200)
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
    last_status = 0.0
    skipped = 0           # frames the reader got that the main loop never used
    print(f"Running on device={DEVICE}, headless={HEADLESS}")

    while True:
        t0 = time.time()
        prev_id = frame_id
        ok, frame, frame_id, frame_time = cam.read(frame_id)
        skipped += max(0, frame_id - prev_id - 1)
        if not ok:
            print("No camera frame")
            break
        t1 = time.time()

        points, prev_center, result = pose_one_frame(pose_model, frame, prev_center)
        history.append((frame_time, points))
        t2 = time.time()

        now = time.time()
        model_says_fall = False
        torso_angle = None   # None = can't see the body well enough to judge
        leg_angle = None     # None = ankles not visible

        window = resample_to_train_fps(history)   # None until 1.16 s of frames exist
        if window is not None:
            # Same cleaning as training: fill short gaps, then quality check
            processed, _, remaining = interpolate_short_gaps(
                window, TRAIN_FPS, max_gap_seconds=MAX_GAP_SECONDS)
            missing = float(remaining[:, CORE_JOINTS].mean())

            if missing <= MAX_MISSING_FRACTION:   # training skipped worse windows
                processed[:, :, :2][remaining] = np.nan
                feats = window_features(processed)
                torso_angle = float(feats[TORSO_IDX])  # 0 = upright, 90 = lying
                if np.isnan(torso_angle):
                    torso_angle = None
                leg_angle = legs_angle(processed)
                x = feats.reshape(1, -1)
                if threshold is not None:
                    fall_col = list(classifier.classes_).index(FALL_LABEL)
                    model_says_fall = classifier.predict_proba(x)[0, fall_col] >= threshold
                else:
                    model_says_fall = classifier.predict(x)[0] == FALL_LABEL

        upright = torso_angle is not None and torso_angle < UPRIGHT_ANGLE
        on_feet = leg_angle is not None and leg_angle < ON_FEET_ANGLE
        # Ankles hidden (behind a bed, out of frame): fall back to the old rule,
        # "not upright = down". Missing a real fall is worse than a false alarm.
        lying = (leg_angle > LYING_ANGLE) if leg_angle is not None else not upright
        t3 = time.time()

        # ---- state machine ----
        if state == "NORMAL":
            if model_says_fall:
                state, check_start, lying_since = "CHECKING", now, None
        elif state == "CHECKING":
            if upright:
                state = "NORMAL"                  # came back up quickly
            elif on_feet and now - check_start > GRACE_SECONDS:
                state = "NORMAL"                  # bent over, but still on feet -> picking something up
            elif now - check_start > MAX_CHECK_SECONDS:
                state = "NORMAL"                  # never settled on the floor
            elif lying:
                lying_since = lying_since or now
                if now - lying_since >= CONFIRM_SECONDS:
                    state, upright_since = "FALLEN", None
                    fallen_since = now
                    alerts.fall(torso_angle=torso_angle)  # once per fall
            else:
                lying_since = None                # in between: wait, don't count
        elif state == "FALLEN":
            if upright or on_feet:
                upright_since = upright_since or now
                if now - upright_since >= RECOVER_SECONDS:
                    state = "NORMAL"              # standing again for a while
                    alerts.recovered(upright_since - fallen_since)
            else:
                upright_since = None              # still down -> keep alarm on

        if window is None:
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
            if now - last_status >= 5:   # + every 5 s: are we keeping up with the camera?
                print(f"  status: loop {fps:.1f} fps | camera {cam.read_fps:.1f} fps | "
                      f"skipped {skipped} old frames (good: we always use the newest)")
                last_status, skipped = now, 0
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
        legs_txt = "?" if leg_angle is None else f"{leg_angle:.0f}"
        cv2.putText(shown, f"{fps:.1f} fps (trained on {TRAIN_FPS})",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("fall detection", shown)
        if TIMING:
            t4 = time.time()
            print(f"wait camera {1000*(t1-t0):4.0f} ms | yolo {1000*(t2-t1):4.0f} ms | "
                  f"forest {1000*(t3-t2):4.0f} ms | blur+draw {1000*(t4-t3):4.0f} ms")
        if cv2.waitKey(1) == ord("q"):
            break

    alerts.stop()
    cam.release()
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()