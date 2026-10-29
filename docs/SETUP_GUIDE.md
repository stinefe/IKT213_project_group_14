# Testing Guide – Fall Detection (IKT213, Group 14)

This guide explains how to set up the repository and run the **offline training
pipeline**, from the raw dataset to a trained and evaluated classifier:

cleaning → indexing → split → pose extraction → windows → training → error analysis

Optional verification tools and experiments are described in section 7.
Live inference on the Jetson Nano is not covered yet.

---

## 1. Repository structure

| Folder | Contents |
|---|---|
| `falldet/` | Shared pipeline code: config, pose preprocessing, features, window building |
| `scripts/data/` | Data preparation steps (cleaning, indexing) |
| `scripts/pose/` | Batch pose extraction |
| `training/` | Train/val/test split, classifier training, error analysis |
| `tools/` | Read-only checks, inspection tools and the annotation tool (not part of the pipeline) |
| `experiments/` | One-off analyses used during development (not part of the pipeline) |
| `data/` | Committed CSV files: `dataset_index.csv`, `splits.csv` |
| `outputs/` | Generated files (not in Git) |
| `models/` | Trained classifier |

Only the steps in section 5 and 6 are needed to reproduce the trained model.

---

## 2. Requirements

| Item | Version / note |
|---|---|
| OS | Tested on Windows |
| Python | 3.12 (3.11 also works). Check with `uv run python --version` |
| Package manager | [uv](https://docs.astral.sh/uv/) (plain `pip` also works, see 4b) |
| Git | Any recent version |
| Internet | Needed on first pose run (downloads `yolo11n-pose.pt`) |

Use the same Python and package versions as the rest of the group. Different
versions can give slightly different pose detections and results.

---

## 3. Get the code and the dataset

### 3.1 Clone the repository
```bash
git clone https://github.com/stinefe/IKT213_project_group_14.git
```

### 3.2 Download the dataset
We use the **Le2i / ImViA Fall Detection Dataset**
(Dubois & Miteran, Université de Bourgogne, DOI: `10.25666/DATAUBFC-2024-04-09`).

The dataset is **not** stored in Git.

### 3.3 Extract the dataset
The download is a zip file containing one zip file per room. **All zip files must
be extracted**, both the outer one and every inner one. The scripts cannot read
videos inside zip files.

### 3.4 Add our own annotations for Office and Lecture_room
The original dataset has no annotations for `Office` and `Lecture_room`. We
annotated these videos ourselves (see section 7.2). The annotation files are
**not** in Git; get them from the group (shared folder) and place them here:

```
FallDataset/Office/Office/Annotation_files/video (1).txt ...
FallDataset/Lecture_room/Lecture room/Annotation_files/video (1).txt ...
```

Each file contains only two numbers: the fall start frame and the fall end
frame, or `0 0` if the video has no fall. A backup of the annotation work is
kept in `own_annotations_log.csv` in the repository.

### 3.5 Place the dataset next to the repository
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

## 4. Set up the Python environment

Run all commands from the repository root.

### 4a. With uv (recommended)
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```
Scripts are then run with `uv run python <script>`.

### 4b. With plain pip
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```
Scripts are then run with `python <script>`.

### 4c. Scripts vs. modules
Files inside the `falldet/` package are run as modules with `-m`, for example
`uv run python -m falldet.build_windows`. This makes `from falldet...` imports
work. Scripts in `scripts/`, `training/` and `tools/` are run with their file
path.

### 4d. Harmless warnings
Reading some Coffee room videos prints many lines like
`[mp3float @ ...] Header missing`. These come from the audio track in the AVI
files and can be ignored; only the video frames are used.

---

## 5. Run the data pipeline

Run the steps **in this order**. Each step lists its input, output and the
expected result.

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1 | `scripts/data/clean_annotations.py` | Home annotation files | `Annotation_files_cleaned/` (in dataset folder) |
| 2 | `scripts/data/build_dataset_index.py` | Videos + annotations | `data/dataset_index.csv` |
| 3 | `training/make_split.py` | Dataset index | `data/splits.csv` |
| 4 | `scripts/pose/<BATCH-POSE-SCRIPT>.py` | Dataset index + videos | `outputs/poses/*.npz`, `outputs/pose_summary.csv` |
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
| train | 124 | <!-- TODO --> | <!-- TODO --> |
| val | 32 | 23 | 9 |
| test (Office) | 33 | <!-- TODO --> | <!-- TODO --> |
| **Total** | **189** | **129** | **60** |

<!-- TODO: fill in from the crosstab printed by make_split.py. -->

The script stops with an `AssertionError` if any video is lost, duplicated,
or placed in more than one split.

> The test set is based on our own annotations. Earlier results used Home_02
> as the test set and are not comparable with results from this split.

### Step 4 – Batch pose extraction
```bash
uv run python scripts/pose/<BATCH-POSE-SCRIPT>.py
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

<!-- TODO: add key numbers from pose_summary.csv (share of frames with one
person / core joints OK, overall and inside the fall interval). -->

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

Frame rate per subset (verified with `tools/check_video_fps.py`):

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
  `Home_01/video (11)` and `Home_02/video (35)` (known; see
  `experiments/inspect_excluded_windows.py`).
- Last line: `Saved-file verification: PASS`

---

## 6. Train and evaluate

| Step | Command | Reads | Writes |
|---|---|---|---|
| 6 | `training/train_baseline.py` | Windows + splits | `models/fall_rf.joblib` |
| 7 | `training/analyse_errors.py` | Model + windows + splits | `outputs/validation_errors.csv`, `outputs/validation_metrics.csv` |

### Step 6 – Train the baseline classifier
```bash
uv run python training/train_baseline.py
```
Converts every window to a feature vector (`falldet/features.py`: hip drop,
hip speed, torso angle, torso angle change, aspect ratio, missing share) and
trains a Random Forest with balanced class weights on the **train** windows.
The test set is not used here.

The script prints, for the **validation** set:
1. Confusion matrix and classification report at the default threshold 0.5
2. The threshold and precision needed to reach 80/85/90/95 % window recall
3. Event-level results for several thresholds: how many fall videos are
   detected by at least one window, and how many videos get a false alarm
   (split into no-fall videos and fall videos before the fall)
4. Feature importances

The model is saved to `models/fall_rf.joblib`, together with the feature names,
window length and scikit-learn version.

**Expected (validation, 1 661 windows, 32 videos):**

Window level at threshold 0.5:

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0.98 | 1.00 | 0.99 |
| Fall | 0.93 | 0.73 | 0.82 |

Confusion matrix: `[[1521 7] [36 97]]`

Window-level threshold trade-off:

| Recall target | Threshold | Precision |
|---:|---:|---:|
| 0.80 | 0.32 | 0.89 |
| 0.85 | 0.28 | 0.86 |
| 0.90 | 0.12 | 0.77 |
| 0.95 | 0.04 | 0.59 |

Event level:

| Threshold | Falls detected | No-fall videos with false alarm | Fall videos with alarm before the fall |
|---:|---:|---:|---:|
| 0.50 | 21/23 | 2/9 | 2 |
| 0.28 | 22/23 | 4/9 | 3 |
| 0.12 | 23/23 | 7/9 | 3 |

Missed at 0.5: `Coffee_room_01/video (30)`, `Lecture_room/video (11)`.
Missed at 0.28: `Lecture_room/video (11)`.

Feature importances: hip_max_speed 0.37, aspect_end 0.28, torso_angle_end
0.19, hip_drop 0.11, torso_angle_change 0.04, missing_frac 0.00.

> **Threshold is not final.** About 0.3 is a reasonable provisional choice.
> The final threshold should be chosen together with the alarm logic
> (temporal confirmation), using the validation set only.

### Step 7 – Error analysis on validation
```bash
uv run python training/analyse_errors.py
```
Loads the saved model, predicts on the validation windows and saves every
misclassified window to `outputs/validation_errors.csv` (video, frames, FPS,
true/predicted label, fall probability, missing fraction and all feature
values), plus window-level metrics to `outputs/validation_metrics.csv`.

Use the CSV for manual review: filter on `error_type`, open the video at
`window_start`–`window_end` and describe what the person does in
`review_notes`. A high `core_missing_fraction` points to pose estimation
problems rather than classifier errors.

> The threshold is set with `THRESHOLD` in the script. Use the same value as
> in the threshold analysis.

### Test set
The Office test set is evaluated **once**, after the threshold and alarm logic
are fixed on the validation set. Do not tune anything after looking at the
test results.

---

## 7. Optional tools and experiments

These scripts are **not** part of the pipeline.

### 7.1 Tools

| Script | Purpose |
|---|---|
| `tools/check_dataset.py` | Read-only check of one subset: folders, video/annotation pairing, decoding, frame counts and order, plus a dataset inventory |
| `tools/extract_pose.py` | Pose test on a single video; saves a skeleton preview video |
| `tools/check_video_fps.py` | Lists the FPS of every indexed video |
| `tools/check_fall_lengths.py` | Lists the length of every annotated fall |
| `tools/annotate_falls.py` | Tool used to create our own fall annotations for Office and Lecture_room |
| `tools/update_zero_zero_review.py` | Old patch script for `visual_review` in the index (superseded) |

### 7.2 Own annotations
Office and Lecture_room were annotated with `tools/annotate_falls.py`. For each
video we recorded the fall start and end frame, or `0 0` if there is no fall,
after watching the whole video. <!-- TODO: describe the rule used for start and
end (e.g. loss of balance → person on the ground) and who annotated. -->

<!-- TODO (recommended): annotate a few videos that already have original Le2i
annotations with the same tool, and report the average difference in frames
for fall_start and fall_end. -->

### 7.3 Experiments

| Script | Purpose |
|---|---|
| `experiments/check_missing_gaps.py` | Distribution of gap lengths in the pose data |
| `experiments/inspect_pose_file.py` | Shows the fields and shapes of one pose file |
| `experiments/inspect_excluded_windows.py` | Shows why windows in two specific videos were kept or excluded |

### 7.4 Dataset check (optional)
```bash
uv run python tools/check_dataset.py
```
Default subset is Home_02 (set with `CHECK_DATASET` in `falldet/config.py`).

**Expected for Home_02:**
- 30 videos and 30 labels found
- 30 × `READ OK`, 30 × `COUNT OK`, 30 × `ORDER OK`
- Videos 31–37 have fall intervals, videos 38–60 have `0, 0`
- Dataset inventory: 190 videos in total, 130 original annotation files

### 7.5 Single-video pose test (optional)
```bash
uv run python tools/extract_pose.py
```
Runs YOLO11n-Pose on one video (default `Home_01/video (1)`) and saves a
skeleton preview and `keypoints.npz` to `outputs/`.

**Expected:**

| Check | Result |
|---|---|
| Frames processed | 264 |
| Keypoints shape | `(264, 17, 3)` |
| One / zero / multiple detections | 261 / 0 / 3 |
| Multiple-detection frames | 124, 125, 127 |

Small differences in detection counts can occur with other package versions.

> **Privacy:** this tool writes video frames with skeletons to disk. Use it on
> the public dataset only, never on own recordings of test persons.

---

## 8. Notes
- **Excluded data:** `Coffee_room_01/video (35)` has conflicting annotations
  and is excluded. All other 189 videos are used.
- **Video labels:** `fall_interval` → Fall, `zero_zero` → Normal. For original
  videos the zero-zero interpretation is supported by visual review; for own
  annotations the whole video was watched.
- **Frame rate:** Home videos are 24 fps, all other subsets 25 fps.
- **Not in Git (by design):** the dataset, cleaned annotation folders, our own
  annotation files (shared separately), `outputs/` and the `.venv` folder.
  <!-- TODO: decide whether models/fall_rf.joblib is committed. -->