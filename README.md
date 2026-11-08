# Real-time Fall Detection (IKT213 – Group 14)

A camera watches a room. When a person falls and stays down, a nurse gets an alert on
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
| `data/` | our annotations (`dataset_index.csv`) and the train / val / test split (`splits.csv`) |
| `models/fall_rf.joblib` | the trained model |
| `docs/SETUP_GUIDE.md` | full step-by-step guide for the training pipeline |
| `Dockerfile`, `.sh` files | running on the Jetson |

---

## 2. Run on a laptop

```bash
git clone https://github.com/stinefe/IKT213_project_group_14.git
cd IKT213_project_group_14
git checkout final-delivery

python -m venv venv
source venv/Scripts/activate        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

python live_detect.py
```

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
   `falldet.env` is listed in both `.gitignore` and `.dockerignore`, so the password
   is never committed or copied into the image.
3. Build the Docker image from the project folder:
   ```bash
   sudo docker build -t falldet .
   ```

`models/fall_rf.joblib` must be trained with **scikit-learn 1.3.2 and numpy 1.24.4**,
the versions inside the container. A model saved with newer versions will not load
(`No module named 'numpy._core'`).

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

To send alerts, start `live_detect.py` with the broker settings:
```bash
DEVICE_ID=room1 MQTT_HOST=<broker address> MQTT_PORT=8883 \
MQTT_USER=<user> MQTT_PASS='<password>' python live_detect.py
```
Never put passwords in the code or in git.

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
annotations for Office and Lecture_room, is in **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**.

Short version:

```bash
python -m scripts.pose.extract_pose_batch    # 1. body keypoints for every video (slow)
python -m falldet.build_windows              # 2. cut into labelled 1-second windows
python -m training.train_baseline            # 3. train + validate → models/fall_rf.joblib
```
Use the `data/splits.csv` that is already in the repo, so the results match ours.

---

## 6. Results

Fall detection per 1-second window, at the chosen threshold **0.4**
(classifier only, without the live confirmation rules):

| | Precision | Recall | F1 | Falls detected |
|---|---:|---:|---:|---:|
| Validation (5 rooms, 32 videos) | 0.89 | 0.89 | 0.89 | 23/23 |
| **Test (Office, a room never seen in training, 33 videos)** | **0.61** | **0.89** | **0.72** | **16/17** |

The model catches falls equally well in a new room (same recall). It gives more
false alarms there, mostly when someone sits down on a chair or picks something
up from the floor. In the live system, the confirmation rules (lying for 2 s with
the legs flat) are designed to filter exactly these cases.

---

## 7. Known limitations

- A fall that ends kneeling can be missed.
- The whole body must be visible to the camera.
- The phone app only receives alerts while it is open (iphone-limitation).

---

## Credits

Le2i Fall Detection Dataset · Ultralytics YOLO11 · scikit-learn · OpenCV · paho-mqtt · EMQX Cloud