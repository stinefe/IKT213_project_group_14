# Testing Guide – Fall Detection (IKT213, Group 14)

This guide explains how to set up and test the current state of the repository.
It covers the **data preparation pipeline** (cleaning, validation, indexing),
a **pose-extraction test**, and the **train/val/test split**.

The classifier, training and Jetson deployment are not implemented yet.

---

## 1. Requirements

| Item | Version / note |
|---|---|
| OS | Tested on Windows |
| Python | 3.12 (3.11 also works) |
| Package manager | [uv](https://docs.astral.sh/uv/) (plain `pip` also works, see 3b) |
| Git | Any recent version |
| Internet | Needed on first pose run (downloads `yolo11n-pose.pt`) |

---

## 2. Get the code and the dataset

### 2.1 Clone the repository
```bash
git clone <REPOSITORY-URL>
```

### 2.2 Download the dataset
We use the **Le2i / ImViA Fall Detection Dataset**
(Dubois & Miteran, Université de Bourgogne, DOI: `10.25666/DATAUBFC-2024-04-09`).

The dataset is **not** stored in Git.

### 2.3 Extract the dataset
The download is a zip file containing one zip file per room. **All zip files must
be extracted**, both the outer one and every inner one. The scripts cannot read
videos inside zip files.

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
    ├── Lecture_room/Lecture room/   (videos only, no annotations)
    └── Office/Office/               (videos only, no annotations)
```

If the dataset is stored elsewhere, change `DATASET_ROOT` in `falldet/config.py`:

```python
DATASET_ROOT = Path(r"C:\path\to\FallDataset")
```

---

## 3. Set up the Python environment

Run all commands from the repository root.

### 3a. With uv (recommended)
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```
Scripts are then run with `uv run python <script>`.

### 3b. With plain pip
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```
Scripts are then run with `python <script>`.

---

## 4. Run the pipeline

Run the steps **in this order**. Each step lists the expected result.

### Step 1 – Clean the Home annotations
```bash
uv run python scripts/data/clean_annotations.py
```
Removes exact duplicate rows from the Home_01 and Home_02 annotation files and
writes cleaned copies to `Annotation_files_cleaned/`. Original files are never
modified.

**Expected:** 60 lines starting with `SAVED:` (30 per Home subset).

| Subset | Files with duplicates | Rows removed |
|---|---:|---:|
| Home_01 | 20 | 61 |
| Home_02 | 27 | 50 |

On a second run the lines say `UNCHANGED:` instead.

### Step 2 – Validate the dataset (read-only)
```bash
uv run python scripts/data/check_dataset.py
```
Checks folders, video/annotation pairing, video decoding, frame counts and
frame order. Default subset is Home_02 (set with `CHECK_DATASET` in
`falldet/config.py`).

**Expected for Home_02:**
- 30 videos and 30 labels found
- 30 × `READ OK`, 30 × `COUNT OK`, 30 × `ORDER OK`
- Videos 31–37 have fall intervals, videos 38–60 have `0, 0`
- Dataset inventory: 190 videos in total, 130 original annotation files
- Office and Lecture_room: `Only AVI videos found.`

### Step 3 – Build the dataset index
```bash
uv run python scripts/data/build_dataset_index.py
```
Creates one row per usable video in `data/dataset_index.csv`, with paths
relative to `DATASET_ROOT`.

**Expected:**

| Subset | Included videos |
|---|---:|
| Coffee_room_01 | 47 |
| Coffee_room_02 | 22 |
| Home_01 | 30 |
| Home_02 | 30 |
| **Total** | **129** |

- With fall interval: **98**
- With zero-zero values: **31**

To confirm the result matches the committed version:
```bash
git diff data/dataset_index.csv
```
No output means the index was reproduced exactly.

### Step 4 – Pose-extraction test
```bash
uv run python <PATH-TO>/extract_pose.py
```
Runs the pretrained YOLO11n-Pose model on one video (default
`Home_01/video (1)`) and saves a skeleton preview and `keypoints.npz` to
`outputs/`. No model is trained in this step.

**Expected:**

| Check | Result |
|---|---|
| Frames processed | 264 |
| Keypoints shape | `(264, 17, 3)` |
| One / zero / multiple detections | 261 / 0 / 3 |
| Multiple-detection frames | 124, 125, 127 |

Small differences in detection counts can occur with other package versions.

### Step 5 – Train/val/test split
```bash
uv run python training/make_split.py
```
Splits the indexed videos at **video level** (never frame or window level).
The whole Home_02 room is held out as the test set. The remaining videos are
split into train and validation, stratified by label (20 % validation,
seed 42). Saved to `data/splits.csv`.

**Expected:**

| split | Fall | Normal | Total |
|---|---:|---:|---:|
| train | 73 | 6 | 79 |
| val | 18 | 2 | 20 |
| test | 7 | 23 | 30 |
| **Total** | **98** | **31** | **129** |

The script stops with an `AssertionError` if any video is lost, duplicated,
or placed in more than one split.

---

## 5. Notes
- **Excluded data:**
  - `Coffee_room_01/video (35)` has conflicting annotations and is excluded.
  - `Office` and `Lecture_room` have no annotation files and are excluded.
    Missing annotations are not treated as "Normal".
- **Video labels:** `fall_interval` → Fall, `zero_zero` → Normal. The
  zero-zero interpretation is a working definition, supported by visual review
  of sampled videos.
- **Not in Git (by design):** the dataset, cleaned annotation folders,
  `outputs/`, model weights and the `.venv` folder.
