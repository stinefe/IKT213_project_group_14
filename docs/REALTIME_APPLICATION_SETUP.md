# Real-time Guide – Fall Detection (IKT213, Group 14)

This guide explains how to run the fall detector **live on a camera** using the
model we have already trained. You do **not** need the dataset and you do not
need to train anything.

For training the model yourself, see `TESTING_GUIDE.md`.

---

## 1. What the application does

`live_detect.py` reads frames from a camera and, for every frame:

1. Runs **YOLO11n-Pose** to find one person and their 17 keypoints
2. Keeps the last ~1.2 seconds of keypoints and resamples them to
   **30 frames at 25 fps**, the same format the model was trained on
3. Fills short gaps and skips low-quality windows, exactly as in training
4. Computes the features and asks the **Random Forest** classifier: fall or not?
5. Confirms the fall with a small state machine (is the person still lying down
   a few seconds later?) before raising an alarm
6. Shows a **blurred** copy of the frame with the skeleton and the status, and
   sends alerts over **MQTT**

The sharp camera frame is only used for pose estimation in memory. It is never
shown, saved or sent anywhere.

---

## 2. What you need

| Item | Note |
|---|---|
| Computer | Windows laptop (tested), or Jetson Nano (see section 9) |
| Camera | Built-in or USB webcam, or Jetson CSI camera |
| Python | 3.12 (3.11 also works) |
| [uv](https://docs.astral.sh/uv/) | Or plain `pip` |
| Internet | Only on the first run (downloads `yolo11n-pose.pt`) |
| MQTT broker | Optional, only if you want alerts on another device (section 8) |

From the repository you need:

| File | What it is |
|---|---|
| `live_detect.py` | The live application |
| `models/fall_rf.joblib` | The trained classifier |
| `falldet/` | Shared code (pose preprocessing, features, face blur, config) |
| `alerts/alerts.py` | MQTT alert client |
| `yolo11n-pose.pt` | Pose model; downloaded automatically if missing |

<!-- TODO: state whether models/fall_rf.joblib is in Git, or where to get it. -->

---

## 3. Set up

```bash
git clone <REPOSITORY-URL>
cd IKT213_project_group_14
uv venv --python 3.12
uv pip install -r requirements.txt
```

Check that the trained model is there:

```bash
ls models/fall_rf.joblib
```

> The classifier must be loaded with (about) the same scikit-learn version it
> was trained with. On startup the script prints what the model file contains;
> the version is stored as `sklearn_version`. If loading fails or gives a
> version warning, install that scikit-learn version.

---

## 4. Place the camera

The model was trained on surveillance-style videos. Detection works best when
the camera looks like the training data:

- **Mounted high** (e.g. top of a shelf or wall corner), **tilted down** at the room
- The **whole body** is visible when the person stands and when they lie on the floor,
  including the feet (ankles are used to tell bending down from lying)
- **One person** in view. With several people, the script follows the person
  closest to the one it tracked last
- Reasonable, even lighting

A camera at **eye level** (e.g. a laptop on a desk) is a different viewpoint
from the training data and gives noticeably worse results. Fine for a quick
test, not for a demo.

---

## 5. Run on a laptop

From the repository root:

```bash
uv run python live_detect.py
```

Press **q** in the video window to quit.

The script must be started from the repository root, because the model path
`models/fall_rf.joblib` is relative.

---

## 6. What you see

The window shows a blurred camera image with the skeleton on top, a status
label and a line with fps and angles.

| Label | Colour | Meaning |
|---|---|---|
| `warming up...` | Grey | Collecting the first ~1.2 s of frames |
| `OK` | Green | No fall |
| `checking... 1.3s` | Yellow | The model thinks a fall started; waiting to see if the person stays down |
| `FALL!` | Red | Fall confirmed; alarm sent |

The second line: `9.8 fps (trained on 25) | torso 12 | legs 8 deg`

- **fps:** frames processed per second
- **torso:** torso angle from vertical (0 = upright, 90 = lying)
- **legs:** hip-to-ankle angle from vertical (0 = on your feet, 90 = flat on the floor);
  `?` when the ankles are not visible

### How a fall is confirmed

```
NORMAL ──model says fall──▶ CHECKING ──lying for 2 s──▶ FALLEN
   ▲                           │                           │
   └── upright / still on feet / 8 s timeout ──┘            │
   └──────────── upright or on feet for 3 s ────────────────┘
```

- **CHECKING → NORMAL** if the person is upright again, or still on their feet
  after 1 s (e.g. bending down to pick something up), or nothing is decided
  within 8 s
- **CHECKING → FALLEN** when the person has been lying for 2 s in a row. One
  alarm is sent per fall
- **FALLEN → NORMAL** when the person has been upright or on their feet for 3 s.
  A "recovered" message is sent

If the ankles are hidden (e.g. behind furniture), "lying" falls back to "torso
not upright". This favours an alarm over a missed fall.

---

## 7. Options

Options are set with environment variables.

| Variable | Default | Effect |
|---|---|---|
| `HEADLESS=1` | off | No window; prints a line each time the state changes |
| `DEVICE` | `0` if a CUDA GPU is found, else `cpu` | Where YOLO runs |
| `CAMERA` | laptop webcam / Jetson CSI | Camera index (`1`) or a stream URL |
| `TIMING=1` | off | Prints milliseconds per step (camera, YOLO, classifier, drawing) |

Git Bash:
```bash
HEADLESS=1 uv run python live_detect.py
CAMERA=1 uv run python live_detect.py
```

PowerShell:
```powershell
$env:HEADLESS="1"; uv run python live_detect.py
```

Headless output looks like:
```
10:42:13 NORMAL   fps=9.6 torso=8
10:42:20 CHECKING fps=9.4 torso=71
10:42:22 FALLEN   fps=9.5 torso=84
```

---

## 8. MQTT alerts

The application sends alerts through `alerts/alerts.py` using the settings in
`falldet/config.py`:

| Setting | Default |
|---|---|
| `MQTT_BROKER` | `localhost` (set to the Jetson's IP to receive alerts elsewhere) |
| `MQTT_PORT` | `1883` |
| `MQTT_TOPIC` | `falldetection/alerts` – full alert as JSON |
| `MQTT_STATUS_TOPIC` | `falldetection/status` – simple `fall` / `ok` (e.g. for an LED) |

Messages:
- **Fall:** once per confirmed fall, with the torso angle
- **Recovered:** when the person is back on their feet, with how long they were down
- **Heartbeat:** about every 10 s with state and fps ("I'm alive")

To watch the messages on the same computer (with Mosquitto installed):
```bash
mosquitto_sub -h localhost -t "falldetection/#" -v
```

<!-- TODO: describe what happens if no broker is running (does the app keep
running without alerts?), and how to start a broker for the demo. -->

The alerts contain status and numbers only, never images.

---

## 9. Run on the Jetson Nano

On the Jetson (ARM, `aarch64`) the script automatically opens the **CSI
camera** at 640×480 through GStreamer. A USB camera or stream can be used
instead with `CAMERA`. YOLO runs on the GPU if CUDA is available.

<!-- TODO: Jetson version (Nano / Orin Nano), JetPack version, how to install
PyTorch/ultralytics/scikit-learn there, or how to run with Docker
(image name, docker run command with camera access and HEADLESS=1). -->

Run headless if no screen is connected:
```bash
HEADLESS=1 python live_detect.py
```

---

## 10. Settings you can tune

These are at the top of `live_detect.py` and only affect the logic **after**
the classifier. They can be changed without retraining.

| Setting | Default | Meaning |
|---|---:|---|
| `UPRIGHT_ANGLE` | 30° | Torso closer to vertical than this = upright |
| `ON_FEET_ANGLE` | 40° | Legs steeper than this = still on your feet |
| `LYING_ANGLE` | 60° | Legs flatter than this = lying or sitting on the floor |
| `GRACE_SECONDS` | 1.0 s | Time to land before the legs are judged |
| `CONFIRM_SECONDS` | 2.0 s | Must be lying this long to confirm a fall |
| `RECOVER_SECONDS` | 3.0 s | Must be upright this long to clear the alarm |
| `MAX_CHECK_SECONDS` | 8 s | Maximum time in CHECKING |

Shorter `CONFIRM_SECONDS` gives faster alarms but more false alarms when
someone sits down on the floor on purpose.

---

## 11. Settings that must match training

These must be the same as when `models/fall_rf.joblib` was trained. **Do not
change them** unless you also rebuild the windows and retrain.

| Setting in `live_detect.py` | Must match |
|---|---|
| `WINDOW = 30` | `WINDOW_LENGTH` in `falldet/build_windows.py` |
| `TRAIN_FPS = 25` | Frame rate of the training videos |
| `MAX_GAP_SECONDS = 0.12` | Same name in `falldet/build_windows.py` |
| `MAX_MISSING_FRACTION = 0.30` | Same name in `falldet/build_windows.py` |
| `CORE_JOINTS` | Same name in `falldet/build_windows.py` |
| `DETECTION_CONF`, `KEYPOINT_CONF`, `NMS_IOU`, `MIN_GOOD_KEYPOINTS` | Batch pose extraction script |
| Camera at **4:3** (640×480) | `IMG_ASPECT = 4/3` in `falldet/features.py` |

**Decision threshold:** if the model file contains a `threshold`, it is used.
Otherwise the classifier's default (0.5) is used. The startup output says
which one.

<!-- TODO: once a threshold is chosen, save it in the model file in
train_baseline.py and note the value here. -->

**After retraining:** copy the new `models/fall_rf.joblib` to the device. No
change to `live_detect.py` is needed as long as the settings above are unchanged.

---

## 12. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `Could not open camera` | Wrong camera index, or another app uses the camera. Try `CAMERA=1` |
| `No camera frame` and exit | Camera disconnected or stream ended |
| Stays on `warming up...` | No frames with timestamps yet; check fps on screen and the camera |
| Very low fps | YOLO on CPU. Use a GPU (`DEVICE=0`), or check with `TIMING=1` which step is slow |
| Never detects a fall | Camera too low or feet out of view; see section 4. Check that torso/legs angles change when you lie down |
| Alarm when sitting on the floor | Expected limitation; increase `CONFIRM_SECONDS` or `LYING_ANGLE` |
| Error loading `fall_rf.joblib` | scikit-learn version differs from training; see section 3 |
| `FileNotFoundError: models/fall_rf.joblib` | Not started from the repository root |

---

## 13. Privacy

- Pose estimation runs on the sharp frame **in memory only**.
- The window shows a **blurred** copy with faces hidden. Faces are found from
  the pose keypoints; when no face keypoints are visible, the top part of the
  person's box is blurred instead.
- Nothing is written to disk and no images are sent over MQTT.
- **Limitation:** blurring needs a detected person. In a frame where YOLO finds
  nobody (e.g. motion blur during a fast fall), that frame is shown unblurred.
  Use `HEADLESS=1` when no picture should be shown at all.
- Recording other people requires their consent.

---

## 14. Known limitations

- Trained on 4 environments from the Le2i dataset plus our own annotations;
  other rooms and camera angles may work worse.
- The live frame rate is usually much lower than 25 fps. The keypoints are
  resampled to 25 fps, but fast movements between real frames are smoothed,
  so a very fast fall can look slower than in the training data.
- Only one person is tracked.
- Sitting or lying down on the floor on purpose can trigger an alarm.