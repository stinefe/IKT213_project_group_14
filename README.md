# Real-time Fall Detection (IKT213 – Group 14)

> **Delivered version:** The submitted implementation is in [final-delivery](https://github.com/stinefe/IKT213_project_group_14/tree/final-delivery). This README on `main` describes that version. Follow the checkout command below before running it; some delivery files are not present on `main`.

This student project is a fall-detection prototype. A camera watches a room. When a person falls and stays down, a nurse gets an alert on
their phone and can press **Acknowledge** to say "I'm on my way".
It runs on an **NVIDIA Jetson Nano**. Faces are blurred on screen for privacy, and no images leave the device.

```
camera → body keypoints (YOLO) → movement features → Random Forest → rules → alert → phone
```

---

## 1. What's in the repo

| Folder / file | What it does |
|---|---|
| `live_detect.py` | the live system: camera → fall detection → alert |
| `falldet/` | shared code used by both training and live detection (features, gap filling, face blur, window building) |
| `alerts/` | sends alerts to the MQTT broker, plus a terminal listener |
| `scripts/` | data preparation and pose extraction (training pipeline steps 1–4) |
| `training/` | split, training, error analysis and the final test (steps 3, 6–8) |
| `data/` | Video index (`dataset_index.csv`), video-level splits (`splits.csv`), and manual annotations in `manual_annotations/` on the delivery branch |
| `models/fall_rf.joblib` | the trained model |
| `docs/SETUP_GUIDE.md` | Full training guide; use the version on `final-delivery` linked below |
| `Dockerfile`, `.sh` files | running on the Jetson |

---

## 2. Run on a laptop

Start with a new clone, or use a separate folder if you already have local work:

```bash
git clone --branch final-delivery https://github.com/stinefe/IKT213_project_group_14.git
cd IKT213_project_group_14
python -m venv venv
```

Activate the environment using the command for your terminal:

| Terminal | Command |
|---|---|
| Windows PowerShell | `.\venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `venv\Scripts\activate.bat` |
| Windows Git Bash | `source venv/Scripts/activate` |
| Linux/macOS shell | `source venv/bin/activate` |

Then run the delivery branch's documented installation:

```bash
python -m pip install -r requirements.txt
python live_detect.py
```

**Environment note:** The laptop requirements pin scikit-learn 1.7.2, while the
Jetson Dockerfile defaults to 1.3.2. The delivery documentation specifies
scikit-learn 1.3.2 and NumPy 1.24.4 for the Jetson model. Use a compatible
model/environment pair; the laptop installation above is not a guarantee that
a model saved in the Jetson environment will load without compatibility issues.
The project package metadata also differs from the Jetson environment.

MQTT settings are needed for phone notifications; see Section 4.

Stand so your **whole body, feet included**, is visible. Press **q** to quit.

| On screen | Meaning |
|---|---|
| 🟢 OK | normal |
| 🟡 checking... | possible fall, checking if the person stays down |
| 🔴 FALL! | fall confirmed, alert sent |
| 🟠 FALL - nurse on the way | the nurse pressed Acknowledge |

---

## 3. Run on the Jetson Nano

### One-time setup

1. Copy the project folder to the Jetson.
2. Create `falldet.env` in the project folder with the MQTT settings:
   ```
   DEVICE_ID=room1
   MQTT_HOST=<broker address>
   MQTT_PORT=8883
   MQTT_USER=<user>
   MQTT_PASS=<password>
   ```
   `falldet.env` is listed in both `.gitignore` and `.dockerignore`, to exclude this local settings file from normal Git additions and Docker builds.
   Do not commit credentials; ignore rules do not remove files already tracked by Git.
3. Build the Docker image from the project folder:
   ```bash
   sudo docker build -t falldet .
   ```

The delivery documentation specifies **scikit-learn 1.3.2 and NumPy 1.24.4**
for the Jetson model. The Dockerfile sets scikit-learn 1.3.2 by default but inherits
NumPy from its base image. Check the actual environment before retraining.
The team encountered `No module named 'numpy._core'` when using an incompatible model.

### Every time

**Terminal 1** – camera stream (leave it running):
```bash
bash jetson_camera_stream.sh
```

**Terminal 2** – fall detection, with a preview window on the Jetson's screen:
```bash
bash run_with_display.sh
```


## 4. Alerts (MQTT)

We use a cloud MQTT broker: **EMQX Cloud Serverless** ([cloud-intl.emqx.com](https://cloud-intl.emqx.com)).
Because the broker is in the cloud, the Jetson and the phone can be on **different networks**:
we tested the Jetson in one city and the phone app in another.

In Bash, start `live_detect.py` with your broker settings (replace the placeholders):
```bash
DEVICE_ID=room1 MQTT_HOST="<broker address>" MQTT_PORT=8883 \
MQTT_USER="<user>" MQTT_PASS="<password>" python live_detect.py
```
In Windows PowerShell, set the variables separately before starting:

```powershell
$env:DEVICE_ID = "room1"
$env:MQTT_HOST = "<broker address>"
$env:MQTT_PORT = "8883"
$env:MQTT_USER = "<user>"
$env:MQTT_PASS = "<password>"
python live_detect.py
```

Never put passwords in the code or in Git. Use your broker's credentials and
configure the phone with the same broker and device topic.

**Phone app (IoT MQTT Panel):** connect with protocol **TCP-SSL**, port **8883**, and add:

| Panel | Topic | Settings |
|---|---|---|
| LED (fall alarm) | `falldet/room1/led` | on `fall`, off `clear` |
| Button (Acknowledge) | `falldet/room1/ack` | payload `{"event": "ack", "by": "nurse"}` |
| Text log | `falldet/+/alert` | JSON on, JsonPath `$.text` |

---

## 5. Train the model yourself (optional)

Needs the **Le2i Fall Detection Dataset** (not included).
The full step-by-step guide, including how to place the dataset and our own
annotations for Office and Lecture_room, is in **[delivery training guide](https://github.com/stinefe/IKT213_project_group_14/blob/final-delivery/docs/SETUP_GUIDE.md)**.

The delivered dataset index contains **189 videos: 129 with fall intervals and
60 without falls**. Office and Lecture_room add 60 manually annotated videos to
the earlier 129-video dataset. Coffee_room_01/video (35) remains excluded because
its annotation does not match the video.

| Split | Videos | Fall videos | No-fall videos |
|---|---:|---:|---:|
| Training | 124 | 89 | 35 |
| Validation | 32 | 23 | 9 |
| Test: Office | 33 | 17 | 16 |

Windows from one video stay in one split. Office is held out from training and
validation. Manual annotation files are committed on `final-delivery` under
[`data/manual_annotations/`](https://github.com/stinefe/IKT213_project_group_14/tree/final-delivery/data/manual_annotations).
The training guide's final note saying they are not in Git is outdated.

The documented preparation output is **10,948 windows: 10,158 Normal and 790 Fall**.
Windows contain 30 frames with a step of 5 frames, approximately 1.2 seconds at
25 FPS or 1.25 seconds at 24 FPS (Home subsets). Short bounded gaps of up to
0.12 seconds are interpolated. Windows with more than 30% remaining missing
shoulder/hip observations are excluded.

A pre-fall window is Normal. A Fall window must overlap the fall by at least
`0.5 × min(30, fall length)` frames. Windows starting after `fall_end` are excluded;
a qualifying window may include the landing after `fall_end`. Other partially
overlapping windows are excluded. Reviewed no-fall videos supply Normal windows.
The quality filter retains Fall windows for 127 of 129 annotated fall videos;
this is data coverage, not classifier recall.

Six features describe hip drop, maximum downward hip movement per frame,
final torso angle, change in torso angle, final body aspect ratio, and final
legs angle. The classifier is a Random Forest with 200 trees and balanced
class weights.

Short version, after placing the videos and configuring `DATASET_ROOT` in
`falldet/config.py`, using the committed index and splits:

```bash
python -m scripts.pose.extract_pose_batch    # 1. body keypoints for every video (slow)
python -m falldet.build_windows              # 2. build labelled 30-frame windows
python -m training.train_baseline            # 3. train + validate → models/fall_rf.joblib
```
Use the `data/splits.csv` that is already in the repo, to preserve the documented video partition. Package versions and generated pose files also affect reproducibility. Training overwrites `models/fall_rf.joblib`.

---

## 6. Results

The delivery guide reports these results at threshold **0.4**, before the live
confirmation rules. Precision, recall and F1 are **window-level** metrics; the
last column counts **fall events represented by at least one positive Fall window**.

| | Precision | Recall | F1 | Fall events detected |
|---|---:|---:|---:|---:|
| Validation (5 rooms, 32 videos) | 0.89 | 0.89 | 0.89 | 23/23 |
| **Test (Office, a room never seen in training, 33 videos)** | **0.61** | **0.89** | **0.72** | **16/17** |

Window recall rounded to 0.89 on both splits in this experiment, but precision
fell to 0.61 in Office. This does not establish equal performance in other rooms.
The documented Office confusion matrix is:

| Actual / predicted | Normal | Fall |
|---|---:|---:|
| Normal | 2,284 | 64 |
| Fall | 12 | 99 |

The test contains 2,459 windows. The classifier missed the fall in Office/video (11).
Twelve of the sixteen no-fall Office videos had at least one false-positive window.
These are classifier predictions, not counts of phone alerts after confirmation.

At threshold 0.5, the guide reports validation Fall precision 0.95 and recall
0.86; those numbers must not be mixed with the 0.4 results above.

The live confirmation logic checks for continued lying for two seconds, with
recovery after three seconds upright. It is intended to reduce false alerts;
the offline classifier metrics do not measure the complete alert system.
The team reports successful MQTT delivery between a Jetson and a phone in
different cities. End-to-end live recall and notification latency need further
measurement.

---

## 7. Known limitations

- A fall that ends kneeling can be missed.
- The whole body must be visible to the camera.
- In the team's tested iPhone setup, the phone app needed to remain open to receive alerts.
- Intentional lying down can also trigger an alert.
- The system follows one person at a time; reliable multi-person operation is not established.
- Staged falls and a single held-out room do not demonstrate performance for all real-world falls.

---

## Development history

The earlier dataset checks, duplicate-row cleanup, pose tests and 6,106-window
experiment are preserved in [Data preparation history](docs/DATA_PREPARATION_HISTORY.md).
Those numbers describe the original 129-video stage, not the delivered 189-video model.

## Credits

Le2i Fall Detection Dataset · Ultralytics YOLO11 · scikit-learn · OpenCV · paho-mqtt · EMQX Cloud