# Training Guide – Fall Detection (IKT213, Group 14)

This guide explains how to reproduce the **training pipeline**, from the raw
dataset to a trained and evaluated classifier:

cleaning → indexing → split → pose extraction → windows → training → error analysis

For cloning the repository, installing the Python packages and running the live
system (laptop or Jetson Nano), see the [README](../README.md).

> **Shortcut:** `data/dataset_index.csv` and `data/splits.csv` are already in Git.
> If you only want to retrain the model, you can **skip steps 1–3** and start at
> step 4. Our own annotation files are then not needed either, because the fall
> frames are already in the index.

---

## 1. Requirements

| Item | Note |
|---|---|
| Python | 3.12 (3.11 also works) |
| Package manager | [uv](https://docs.astral.sh/uv/) or plain `pip` (see README) |
| Internet | Needed on the first pose run (downloads `yolo11n-pose.pt`) |
| Disk / time | Pose extraction on all 189 videos takes a long time on CPU |

Use the same Python and package versions as the rest of the group. Different
versions can give slightly different pose detections and results.

**Scripts vs. modules:** files inside the `falldet/` package are run as modules
with `-m` (for example `uv run python -m falldet.build_windows`), so that the
`from falldet...` imports work. Scripts in `scripts/` and `training/` are run with
their file path.

**Harmless warnings:** some Coffee room videos print many lines like
`[mp3float @ ...] Header missing`. These come from the audio track in the AVI
files and can be ignored; only the video frames are used.

---

## 2. Get the dataset

### 2.1 Download
We use the **Le2i / ImViA Fall Detection Dataset**
(Dubois & Miteran, Université de Bourgogne, DOI: `10.25666/DATAUBFC-2024-04-09`).
The dataset is **not** stored in Git.

### 2.2 Extract
The download is a zip file containing one zip file per room. **All zip files must
be extracted**, both the outer one and every inner one. The scripts cannot read
videos inside zip files.

### 2.3 Our own annotations for Office and Lecture_room
The original dataset has no annotations for `Office` and `Lecture_room`, so we
annotated these videos ourselves. For each video we watched the whole video and
recorded the fall start and end frame, or `0 0` if there is no fall.
<!-- TODO: describe the rule used for start and end (e.g. loss of balance →
person on the ground) -->

The annotation files are **not** in Git; get them from the group (shared folder)
and place them here:

```
FallDataset/Office/Office/Annotation_files/video (1).txt ...
FallDataset/Lecture_room/Lecture room/Annotation_files/video (1).txt ...
```

Each file contains only two numbers: the fall start frame and the fall end frame.
<!-- TODO: own_annotations_log.csv (backup of the annotation work) is not in Git
yet: add it, or remove this note. -->

### 2.4 Place the dataset next to the repository
The default configuration expects `FallDataset` **beside** the project folder,
not inside it:

```
<parent folder>/
├── IKT213_project_group_14/        ← the repository
└── FallDataset/
    ├── Coffee_room_01/Coffee_room_01/Videos/, Annotation_files/
    ├── Coffee_room_02/Coffee_room_02/Videos/, Annotations_files/
    ├── Home_01/Home_01/Videos/, Annotation_files/
    ├── Home_02/Home_02/Videos/, Annotation_files/
    ├── Lecture_room/Lecture room/   videos + own Annotation_files/
    └── Office/Office/               videos + own Annotation_files/
```

`Annotation_files_cleaned/` for Home_01 and Home_02 is created by step 1.
Note that the inner Lecture room folder name contains a space.

If the dataset is stored elsewhere, change `DATASET_ROOT` in `falldet/config.py`:

```python
DATASET_ROOT = Path(r"C:\path\to\FallDataset")
```

---

## 3. Run the data pipeline

Run the steps **in this order**, from the repository root.

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1 | `scripts/data/clean_annotations.py` | Home annotation files | `Annotation_files_cleaned/` (in dataset folder) |
| 2 | `scripts/data/build_dataset_index.py` | Videos + annotations | `data/dataset_index.csv` |
| 3 | `training/make_split.py` | Dataset index | `data/splits.csv` |
| 4 | `scripts/pose/extract_pose_batch.py` | Dataset index + videos | `outputs/poses/*.npz`, `outputs/pose_summary.csv` |
| 5 | `-m falldet.build_windows` | Pose files + index | `outputs/windows.npz` |

### Step 1 – Clean the Home annotations
```bash
uv run python scripts/data/clean_annotations.py
```
Removes exact duplicate rows from the Home_01 and Home_02 annotation files and
writes cleaned copies to `Annotation_files_cleaned/`. Original files are never
modified. Rows with the same frame number but different values are not
"fixed"; the script stops with an error instead.

**Expected:** 60 lines starting with `SAVED:` (30 per Home subset).

| Subset | Files with duplicates | Rows removed |
|---|---:|---:|
| Home_01 | 20 | 61 |
| Home_02 | 27 | 50 |

On a second run the lines say `UNCHANGED:` instead.

### Step 2 – Build the dataset index
```bash
uv run python scripts/data/build_dataset_index.py
```
Validates every video/annotation pair and creates one row per usable video in
`data/dataset_index.csv`, with paths relative to `DATASET_ROOT`.

- **Original subsets** (Coffee room, Home): checks frame count, frame order and
  fall interval against the full Le2i annotation files.
- **Own annotations** (Office, Lecture_room): checks that each file has exactly
  two values and a valid fall interval.

The column `annotation_source` is `original` or `own`. The script stops if
anything is invalid or if the total is not 189.

**Expected:**

| Subset | Source | Included videos |
|---|---|---:|
| Coffee_room_01 | original | 47 |
| Coffee_room_02 | original | 22 |
| Home_01 | original | 30 |
| Home_02 | original | 30 |
| Lecture_room | own | 27 |
| Office | own | 33 |
| **Total** | | **189** |

- With fall interval: **129**
- With zero-zero values: **60**

To confirm the result matches the committed version:
```bash
git diff data/dataset_index.csv
```
No output means the index was reproduced exactly.

### Step 3 – Train/val/test split
```bash
uv run python training/make_split.py
```
Splits the indexed videos at **video level** (never frame or window level).
The whole **Office** room is held out as the test set, so the test measures
performance in an environment the model has never seen. The remaining five
subsets are split into train and validation, stratified by label (20 %
validation, seed 42). Saved to `data/splits.csv`.

**Expected:**

| split | Videos | Fall | Normal |
|---|---:|---:|---:|
| train | 124 | 89 | 35 |
| val | 32 | 23 | 9 |
| test (Office) | 33 | 17 | 16 |
| **Total** | **189** | **129** | **60** |

The script stops with an `AssertionError` if any video is lost, duplicated,
or placed in more than one split.

### Step 4 – Batch pose extraction
```bash
uv run python scripts/pose/extract_pose_batch.py
```
Runs the pretrained YOLO11n-Pose model on every indexed video and saves one
`.npz` file per video to `outputs/poses/`, plus quality numbers per video in
`outputs/pose_summary.csv`. No model is trained in this step.

Settings (in the script): detection confidence 0.1, NMS IoU 0.5, keypoint
confidence 0.5, at least 5 confident joints per detection, nearest-person
tracking when several people are detected. Keypoints are stored normalized
(0–1); uncertain joints are stored as NaN.

**Runtime:** the script runs on CPU (`device="cpu"`) and processing all 189
videos takes a long time. Set `LIMIT = 3` for a quick test first. Already
processed videos are skipped on later runs. With an NVIDIA GPU and CUDA-enabled
PyTorch, `device=0` is much faster.

> **Important:** the skip/resume logic does not check whether settings have
> changed. If you change any pose setting, delete `outputs/poses/` before
> running again, otherwise old and new pose files are mixed.

**Expected:** `Done. 189 ok, 0 failed.`

**Pose quality** (from `outputs/pose_summary.csv`, 189 videos, 75 680 frames):

| | Share of frames |
|---|---:|
| Exactly one person detected | 87.4 % |
| All core joints (shoulders + hips) visible | 87.8 % |
| All core joints visible **inside the fall** (mean per video) | 89.2 % |

The pose is found reliably, also during the fall itself, which is the hardest
moment (fast movement, unusual body position).

### Step 5 – Build labelled windows
```bash
uv run python -m falldet.build_windows
```
Cuts each pose sequence into windows of 30 frames (step 5), labels them and
removes low-quality windows. Saved to `outputs/windows.npz`. The file is
reopened and verified after saving.

Labelling rules:
- Window entirely before the fall → **Normal**
- Window overlaps the fall by at least `0.5 × min(30, fall length)` frames → **Fall**
- Window starting after the fall ends → **excluded** (post-fall)
- Partly overlapping window that fails the Fall rule → **excluded**
- Zero-zero video → all windows **Normal**. Allowed only for manually reviewed
  original videos or videos with `annotation_source = own`.

Quality rule: short gaps (≤ 0.12 s) are interpolated per joint inside each
window; windows where more than 30 % of the core joints (shoulders and hips)
are still missing are removed.

Frame rate per subset:

| FPS | Subsets | Videos |
|---:|---|---:|
| 24 | Home_01, Home_02 | 60 |
| 25 | Coffee_room_01, Coffee_room_02, Lecture_room, Office | 129 |

**Expected:**

| | Candidates | Kept | Removed by quality |
|---|---:|---:|---:|
| Normal | 11 333 | 10 158 | 1 175 |
| Fall | 860 | 790 | 70 |

- Windows excluded by labelling rules: **1 926**
- Total windows saved: **10 948** (`X` shape `(10948, 30, 17, 3)`)
- Fall share: **7.2 %**
- Falls represented: **127/129**. No retained Fall window for
  `Home_01/video (11)` and `Home_02/video (35)` (known: all their fall windows
  fail the quality rule).
- Last line: `Saved-file verification: PASS`

---

## 4. Train and evaluate

| Step | Command | Reads | Writes |
|---|---|---|---|
| 6 | `training/train_baseline.py` | Windows + splits | `models/fall_rf.joblib` |
| 7 | `training/analyse_errors.py` | Model + windows + splits | `outputs/validation_errors.csv`, `outputs/validation_metrics.csv` |
| 8 | `training/evaluate_test.py` | Model + windows + splits | (prints only) |

### Step 6 – Train the classifier
```bash
uv run python training/train_baseline.py
```
Converts every window to a feature vector (`falldet/features.py`: hip drop,
hip max speed, torso angle at the end, torso angle change, body aspect ratio at
the end, legs angle at the end) and trains a Random Forest (200 trees,
`min_samples_leaf=5`, balanced class weights) on the **train** windows. The test
set is not used here.

The script prints, for the **validation** set:
1. Confusion matrix and classification report at the default threshold 0.5
2. The threshold and precision needed to reach 80/85/90/95 % window recall
3. Event-level results for several thresholds: how many fall videos are
   detected by at least one window, and how many videos get a false alarm
   (split into no-fall videos and fall videos before the fall)
4. Feature importances

The model is saved to `models/fall_rf.joblib`, together with the feature names,
window length, scikit-learn version and the chosen decision **threshold (0.4)**.
`live_detect.py` reads the threshold from this file.

> **Jetson:** the Docker image on the Jetson has scikit-learn 1.3.2. Train the
> final model with the same version, otherwise it may not load there.

**Expected (validation, 1 661 windows, 32 videos):**

Window level at the default threshold 0.5 (what the script prints first):

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0.99 | 1.00 | 0.99 |
| Fall | 0.95 | 0.86 | 0.91 |

Confusion matrix: `[[1522 6] [18 115]]`

Event level:

| Threshold | Falls detected | No-fall videos with false alarm | Fall videos with alarm before the fall |
|---:|---:|---:|---:|
| 0.5 | 22/23 | 2 | 3 |
| **0.4** | **23/23** | 4 | 3 |
| 0.3 | 23/23 | 5 | 4 |
| 0.2 | 23/23 | 7 | 4 |

Feature importances: hip_max_speed 0.36, aspect_end 0.27, torso_angle_end 0.18,
legs_angle_end 0.09, hip_drop 0.09, torso_angle_change 0.02.

### Step 7 – Error analysis on validation
```bash
python -m training.analyse_errors
```
Loads the saved model, predicts on the validation windows and saves every
misclassified window to `outputs/validation_errors.csv` (video, frames, FPS,
true/predicted label, fall probability, missing fraction and all feature
values), plus window-level metrics to `outputs/validation_metrics.csv`.

Use the CSV for manual review: filter on `error_type`, open the video at
`window_start`–`window_end` and describe what the person does in
`review_notes`. A high `core_missing_fraction` points to pose estimation
problems rather than classifier errors.

> The script uses the threshold saved in the model file (0.4), the same as the
> live system. The CSV also includes `fall_probability` for each wrong window.

**Expected (threshold 0.4):** confusion matrix `[[1514 14] [14 119]]`,
14 false negatives and 14 false positives in 11 videos.
Fall precision 0.89, recall 0.89, F1 0.89.

### Step 8 – Final test (run once)
```bash
python -m training.evaluate_test
```
Evaluates the saved model **once** on the Office test set (a room never seen in
training), at the threshold stored in the model file. Nothing is tuned or saved.
Do not change any setting after looking at these results.

**Result (threshold 0.4, classifier only, without the live confirmation rules):**

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| Fall (per window, 2 459 windows) | 0.61 | 0.89 | 0.72 |

- Falls detected: **16/17** (missed: `Office/video (11)`)
- No-fall videos with at least one false alarm window: **12/16**, mostly
  sitting down on a chair and picking something up from the floor

---

## 5. Notes
- **Excluded data:** `Coffee_room_01/video (35)` has conflicting annotations
  and is excluded. All other 189 videos are used.
- **Video labels:** `fall_interval` → Fall, `zero_zero` → Normal. For original
  videos the zero-zero interpretation is supported by visual review; for own
  annotations the whole video was watched.
- **Frame rate:** Home videos are 24 fps, all other subsets 25 fps.
- **In Git:** the code, `data/dataset_index.csv`, `data/splits.csv` and the
  trained model `models/fall_rf.joblib`.
- **Not in Git (by design):** the dataset, cleaned annotation folders, our own
  annotation files (shared separately), `outputs/` and the virtual environment.