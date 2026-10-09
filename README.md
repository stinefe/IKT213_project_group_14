# Fall Detection: Dataset Preparation, Pose Extraction, and Training Windows

  
**Current stage (9 October 2026):** Dataset validation, annotation cleanup, pose extraction for 129 videos, and labelled-window generation completed. The current window output contains 6,106 examples. Issue #6 documentation and code are ready for group review; classifier performance and Jetson parity are not established by these checks.

This README records the work completed from the first dataset check through pose extraction and labelled training-window generation. It explains what each script does, which files were changed, what the results mean, and what remains to be done.

## 1. Project overview

The planned system uses camera video, YOLO11n Pose body keypoints, movement features across frames, and a lightweight Fall/Normal classifier. The intended deployment device is a Jetson Orin Nano.

| Stage | Current status |
| --- | --- |
| Locate dataset folders | Completed |
| Validate the four annotated subsets | Completed for the checks described below |
| Clean exact duplicate Home annotation rows | Completed; originals preserved |
| Create the dataset index | Completed: 129 videos |
| Review videos with `0, 0` fall values | Completed: all 31 manually reviewed; no falls reported |
| Extract and inspect Coffee_room_01/video (1) poses | Completed |
| Test pose extraction on Home_01/video (1) | Completed |
| Extract poses for all included videos | Completed: 129 pose files |
| Implement window labels and missing-data handling | Implemented and tested; see Section 16 for current rules and review requirements |
| Save labelled training windows | Completed: 5,530 Normal and 576 Fall windows |
| Create data splits and temporal features | Pending |
| Train and evaluate a fall classifier | Pending |
| Deploy on Jetson Orin Nano | Pending |

**Completed checks establish file consistency and frame alignment. They do not establish that every annotation or predicted joint is visually correct.**

## 2. Environment and project files

### Working environment

| Item | Configuration used |
| --- | --- |
| Operating system | Windows |
| IDE | PyCharm |
| Anaconda environment | `ikt213` |
| Python interpreter | `C:\Users\haifa\anaconda3\envs\ikt213\python.exe` |
| Current project directory used for issue #6 | `C:\Users\haifa\Desktop\IKT213_project_group_14` |
| Dataset root | `C:\Users\haifa\Desktop\FallDataset` |
| Libraries used | OpenCV, NumPy, Ultralytics, and Python's standard library |



### Scripts and outputs

| File or folder | Purpose |
| --- | --- |
| `scripts/data/check_dataset.py` | Read-only checks of folders, paired files, video decoding, annotation counts, frame order, and duplicates. Does not modify files. |
| `scripts/data/clean_annotations.py` | Removes exact duplicate annotation rows and saves validated copies in separate cleaned folders, preserving the originals. |
| `falldet/config.py` | Stores shared settings, including the dataset root, selected subset, pose-test video, and output paths. |
| `scripts/data/build_dataset_index.py` | Validate selected annotations and create one CSV row per included video. |
| `scripts/pose/extract_pose.py` | Run pose inference on one indexed video and save numerical keypoints and a preview. |
| `data/dataset_index.csv` | Index of 129 included videos and their selected annotations. |
| `Annotation_files_cleaned` | Separate cleaned annotation folder inside each Home subset. |
| `outputs/pose_check_coffee01_video01` | Output folder for the completed Coffee pose test. |
| `keypoints.npz` | Compressed NumPy file containing frame-aligned keypoints and metadata. |
| `scripts/pose/extract_pose_batch.py` | Extract pose files for the indexed videos. |
| `outputs/poses/` | The 129 per-video pose files used for window generation. |
| `pose_preprocessing.py` | Shared short-gap interpolation function; preserves confidence values. |
| `test_actual_interpolation.py` | Checks interpolation masks, coordinates, and unchanged inputs on real windows. |
| `build_windows.py` | Labels, interpolates, filters, and saves training windows. |
| `outputs/windows.npz` | Generated window arrays and metadata; excluded from Git. |
| `README.md` | This project progress record. |

## 3. Dataset inventory

The extracted local dataset contains **190 AVI videos and 130 original annotation TXT files**. These are the counts observed on this computer.

| Subset | Videos found | Original annotation files | Videos in current index |
| --- | ---: | ---: | ---: |
| Coffee_room_01 | 48 | 48 | 47 |
| Coffee_room_02 | 22 | 22 | 22 |
| Home_01 | 30 | 30 | 30 |
| Home_02 | 30 | 30 | 30 |
| Lecture_room | 27 | 0 found | 0 |
| Office | 33 | 0 found | 0 |
| **Total** | **190** | **130** | **129** |

### Folder locations

Each base folder below is relative to `FallDataset`. The final two columns identify directories inside that base folder.

| Subset | Base folder | Video directory | Original annotation directory |
| --- | --- | --- | --- |
| Coffee_room_01 | `Coffee_room_01/Coffee_room_01` | `Videos` | `Annotation_files` |
| Coffee_room_02 | `Coffee_room_02/Coffee_room_02` | `Videos` | `Annotations_files` |
| Home_01 | `Home_01/Home_01` | `Videos` | `Annotation_files` |
| Home_02 | `Home_02/Home_02` | `Videos` | `Annotation_files` |
| Lecture_room | `Lecture_room/Lecture room` | Directly in base folder | None found |
| Office | `Office/Office` | Directly in base folder | None found |

Coffee_room_02 uses **`Annotations_files`**, with an extra `s`.

Office and Lecture_room were missed by the initial search for folders named `Videos`. A recursive search for AVI files located them. Searches for TXT files and then all non-AVI files inside those two extracted subsets found only videos. We have not established whether their annotations are available elsewhere.

Their **60 videos remain unchanged and outside the index**. Missing annotations must not be treated as Normal labels.

## 4. Dataset checks: Steps 1–16

`scripts/data/check_dataset.py` was built incrementally and run separately for each annotated subset. Steps 11–14 search the wider dataset root; the other steps use the selected subset.

| Step | What the code does | Why it is needed |
| ---: | --- | --- |
| 1 | Defines `dataset` using `Path`. | Selects the subset to inspect. |
| 2 | Builds `videos_folder` and `labels_folder`. | Locates its videos and annotations. |
| 3 | Calls `is_dir()` on both folders. | Confirms that both paths exist. |
| 4 | Counts AVI and TXT files. | Checks the subset inventory. |
| 5 | Matches video and annotation stems in both directions. | Detects missing paired files. |
| 6 | Reads every video frame with OpenCV. | Compares decoded frames with `CAP_PROP_FRAME_COUNT`. |
| 7 | Counts six-column annotation rows. | Checks for one annotation row per video frame. |
| 8 | Extracts frame numbers and compares them with `1, 2, ..., N`. | Detects missing, repeated, or out-of-order frame IDs. |
| 9 | Finds and prints single-value lines wherever they occur. | Inspects fall start/end values without assuming they are the first two lines. |
| 10 | Lists codes found in the second column. | Inspects values without inventing their meanings. |
| 11 | Searches for folders named `Videos`. | Locates subsets and handles both annotation-folder spellings. |
| 12 | Searches recursively for AVI files. | Locates videos stored outside `Videos` directories. |
| 13 | Lists folders containing TXT files. | Locates annotations and distinguishes them from README files. |
| 14 | Searches Office and Lecture_room for non-AVI files. | Checks for possible annotations in another format. |
| 15 | Removes exact duplicate frame rows in memory and rechecks the sequence. | Determines whether duplicate rows explain a mismatch; changes no files. |
| 16 | Cleanup now runs separately in `scripts/data/clean_annotations.py`. | Saves validated cleaned copies while preserving originals. |

Steps 1–16 describe how the checking workflow developed. In the current version, `scripts/data/check_dataset.py` performs read-only checks.

### Path errors corrected during checking

Some early runs printed `False` for both folders and zero files. The configured path contained an extra `C` directory and later an incomplete folder name, `Coffee_room_0`.

We printed `__file__` and the exact video/label paths, corrected the spelling, and confirmed the expected counts before interpreting the checks.

**Zero files checked is not a pass.** Also, `exit code 0` only means the script completed without an unhandled exception; printed validation warnings must still be reviewed.

## 5. Annotation format and parsing decisions

The supplied dataset README describes fall start/end frames and per-frame bounding-box information. The parser uses the observed line structure as follows:

| Content | How it is handled |
| --- | --- |
| Blank line | Ignored during checking. |
| Single-value line | Collected as a candidate fall start/end value. |
| Six comma-separated values | Treated as a frame annotation row. |
| First value of a frame row | Parsed as the frame number. |
| Second value of a frame row | Recorded as an annotation code; meaning remains unconfirmed. |
| Remaining frame-row values | Preserved; exact bounding-box convention remains unconfirmed. |
| UTF-8 byte-order mark | Handled by reading with `utf-8-sig`. |

Codes **1–8** appeared across the inspected data. We did not assume that code 1 means Normal or code 2 means Fall.

The early checks select rows by column count. The later index script also rejects nonblank lines that have neither one nor six columns. Neither script is a complete semantic validator of the bounding boxes or code values.

## 6. Problems found and actions taken

| Dataset/file | Problem | Action | Current result |
| --- | --- | --- | --- |
| Coffee_room_01/video (26) | Two metadata values were placed between frame rows. | Changed parsing to inspect all lines. | All 278 frame rows pass; original annotation retained. |
| Coffee_room_01/video (35) | 471 annotation rows for 231 video frames, with conflicting repeated sequences. | Preserved originals and excluded this pair from the index. | Still unresolved. |
| Home_01 | 20 annotations contained exact duplicate frame rows. | Removed redundant copies into a separate cleaned folder. | All 30 cleaned files pass count/order checks. |
| Home_02 | 27 annotations contained exact duplicate frame rows. | Applied the same validated cleanup. | All 30 cleaned files pass count/order checks. |
| Office and Lecture_room | No annotation files found in extracted folders. | Kept videos outside the supervised-data index. | Annotation source or manual labelling still needed. |

### Coffee_room_01/video (26)

| Detail | Finding |
| --- | --- |
| Video frames | 278 |
| Frame annotation rows | 278, numbered 1–278 |
| Fall values | Start 197; end 227 |
| Location of the two metadata lines | Between frame rows 218 and 219 |
| Earlier parsing error | Skipping the first two lines discarded actual frame rows 1 and 2. |
| Fix | Select six-column rows from the entire file. |
| Visual check | Sampled frames supported the reported fall interval. |

No annotation frame rows were deleted or renumbered for this file.

### Coffee_room_01/video (35)

This readable video has 231 frames, but its annotation contains 471 rows with repeated/reset frame numbers and conflicting box values. Keeping the first row for each frame ID could select incorrect annotations, so no such repair was applied.

The exclusion is scoped to the subset. Home_02/video (35) is a different video and was checked normally:

```python
if dataset.name == "Coffee_room_01" and label.name == "video (35).txt":
    continue
```

### Home annotation cleanup

| Subset | Files with duplicates | Extra rows removed | Cleaned files saved | Verification after saving |
| --- | ---: | ---: | ---: | --- |
| Home_01 | 20 | 61 | 30 | 30 count checks and 30 order checks passed |
| Home_02 | 27 | 50 | 30 | 30 count checks and 30 order checks passed |

For example, Home_01/video (2) has 240 video frames and originally had 245 frame annotation rows. Its frame-240 row occurred six times with identical values. Removing five copies restored the sequence 1–240.

The cleanup compares **all six values, including the frame number**, after trimming surrounding whitespace. It keeps the first occurrence and preserves row order. It does not merge conflicting rows just because they share a frame ID. A cleaned file is saved only when its remaining frame IDs match the full expected video sequence.

| Data | Storage/action |
| --- | --- |
| Original Home annotations | Kept in each original `Annotation_files` directory. |
| Cleaned Home_01 annotations | `Home_01/Home_01/Annotation_files_cleaned` |
| Cleaned Home_02 annotations | `Home_02/Home_02/Annotation_files_cleaned` |
| Fall metadata and other lines | Preserved by the cleaning step. |
| Original videos | Unchanged. |

Files without duplicates were copied too, giving each cleaned directory a complete set of 30 files. Original and cleaned copies must not be counted as separate data examples.

## 7. Reviewing zero-zero fall values

Some annotations contain `0, 0` instead of a positive fall interval. Two examples were reviewed using sampled frames:

| Video | Observed activity | Review outcome |
| --- | --- | --- |
| Home_02/video (38) | Standing, walking to a stool, sitting, and remaining seated. | No fall seen in sampled frames. |
| Coffee_room_02/video (61) | Walking, bending, sitting, leaning back, standing, and sitting elsewhere. | No fall seen in sampled frames. |

These were the initial sampled-frame checks. Subsequently, all 31 indexed zero-zero videos were manually reviewed and reported to contain no falls:

| Subset | Reviewed videos | Count |
| --- | --- | ---: |
| Home_02 | 38–60 | 23 |
| Coffee_room_02 | 61, 63, 65, 66, 67, 68, 69, 70 | 8 |
| **Total** | | **31** |

Their windows can be labelled Normal subject to pose-quality filtering. Original fall values remain `0, 0`. The builder uses an explicit list of these reviewed IDs; a new unreviewed zero-zero video is rejected. This review does not assign labels to Office or Lecture_room.

## 8. Building the dataset index

`scripts/data/build_dataset_index.py` uses original Coffee annotations, cleaned Home annotations, and the scoped exclusion of Coffee_room_01/video (35).

### Checks before saving

| Check | Requirement |
| --- | --- |
| Directories | Video and selected annotation folders exist. |
| Video/annotation pairing | Each included video has its annotation. |
| Video metadata | OpenCV opens the video and reports a positive frame count. |
| Annotation row structure | Each nonblank line has one or six columns. |
| Annotation frame IDs | Exactly `1, 2, ..., frame_count` in order. |
| Fall metadata count | Exactly two integer values. |
| Fall metadata range | Either `0, 0` or `1 <= start <= end <= frame_count`. |
| Included video count | Exactly 129. |

The index script reads video metadata. Full-frame decoding was already performed in Step 6 of `scripts/data/check_dataset.py`.

### Final index results

| Category | Videos |
| --- | ---: |
| Coffee_room_01 included | 47 |
| Coffee_room_02 included | 22 |
| Home_01 included | 30 |
| Home_02 included | 30 |
| **Total indexed** | **129** |
| With positive fall intervals | 98 |
| With zero-zero values | 31 |

The last two rows partition the 129 videos. These are video-level counts, not counts of training frames or temporal windows.

### CSV columns

| Column | Meaning |
| --- | --- |
| `video_id` | Subset-qualified ID, such as `Home_01/video (1)`. |
| `dataset` | Subset name. |
| `video_path` | Path to the video relative to `DATASET_ROOT`. |
| `annotation_path` | Path to the selected annotation relative to `DATASET_ROOT`. |
| `frame_count` | Frame count reported by OpenCV. |
| `fall_start` | First fall metadata value. |
| `fall_end` | Second fall metadata value. |
| `annotation_status` | `fall_interval` or `zero_zero`. |
| `visual_review` | Review metadata. The initial index recorded two sampled-frame reviews; later manual reviews are documented in Section 7. Check the current CSV for stored status values. |

The initial review column was limited: inspection of Coffee_room_01/video (26) and subsequent full zero-zero review are documented here; do not assume that rebuilding the initial index automatically records these later reviews. Subset-qualified IDs prevent confusion between videos with the same filename.

The current configured index location is `data/dataset_index.csv`, relative to the project root. Rebuilding it overwrites that output. Video and annotation paths are relative to `DATASET_ROOT`, which is configured in `falldet/config.py`. Teammates can change the dataset root without rebuilding the CSV, provided the dataset's internal folder structure stays the same.

## 9. First pose-extraction test

`scripts/pose/extract_pose.py` was tested on **Coffee_room_01/video (1)**. It uses a pretrained model for inference; no classifier or pose model was trained during this test.

### Extraction process

| Step | Action |
| ---: | --- |
| 1 | Find the selected video in `data/dataset_index.csv`. |
| 2 | Load `yolo11n-pose.pt`. |
| 3 | Process every video frame and save a skeleton preview. |
| 4 | Record the number of person detections in each frame. |
| 5 | Extract 17 keypoints when exactly one person is detected. |
| 6 | Set low-confidence keypoint coordinates to `NaN`. |
| 7 | Keep every frame in the output, even when its pose is missing. |
| 8 | Verify the processed frame count and save `keypoints.npz`. |

### Settings

| Setting | Value |
| --- | --- |
| Model | `yolo11n-pose.pt` |
| Processing device | CPU |
| Inference image-size setting | `640` |
| Detection confidence threshold | `0.25` |
| Keypoint confidence threshold | `0.5` |
| Video frame stride | `1` |
| Output directory | `outputs/pose_check_coffee01_video01` |

Model preprocessing uses the image-size setting. The original dataset videos were not resized or rewritten.

### Saved NPZ contents

| Field | Meaning |
| --- | --- |
| `video_id` | Source video's subset-qualified identifier. |
| `frame_numbers` | Original one-based frame numbers. |
| `keypoints` | Array of normalized x, normalized y, and confidence for 17 joints per frame. |
| `person_counts` | Number of person detections per frame. |
| `keypoint_confidence_threshold` | Threshold used to filter coordinates: `0.5`. |

The array shape is **`(157, 17, 3)`**: 157 frames, 17 joints, and three values per joint.

X is normalized by image width and y by image height. Later calculations of angles or comparable x/y distances need to account for the original image dimensions.

### Missing-data rules

| Situation | Saved values |
| --- | --- |
| Exactly one detection and joint confidence at least 0.5 | Coordinates and confidence retained. |
| Exactly one detection and joint confidence below 0.5 | Coordinates set to `NaN`; confidence retained. |
| No person detection | All keypoint entries for that frame set to `NaN`. |
| Multiple person detections | All keypoint entries for that frame set to `NaN` for this initial test. |

`NaN` represents missing or filtered data. It must not be treated as coordinate zero. Missing frames remain in the array to preserve alignment with the annotations.

## 10. Pose-test results and review

| Measurement | Result |
| --- | --- |
| Frames processed | 157 |
| Frames with exactly one detection | 151 |
| Frames with no detection | 5: frames 1, 2, 6, 7, 8 |
| Frames with multiple detections | 1: frame 11 |
| Saved keypoint shape | `(157, 17, 3)` |
| Frame order preserved | Yes, 1–157 |
| Frames with both shoulders and both hips available | 140/157 |

In sampled preview frames, the skeleton generally followed the person during walking, falling, and lying down. At the right edge near the start, two overlapping detections appeared to represent the same entering person.

The preview is Ultralytics' rendering. It can display detections or joints that our numerical extraction subsequently filters out.

### Keypoints during the annotated fall

The fall interval is **frames 48–80 inclusive**, covering 33 frames.

| Joint group | Frames with both joints meeting the confidence threshold |
| --- | ---: |
| Shoulders | 33/33 |
| Hips | 33/33 |
| Knees | 33/33 |
| Ankles | 33/33 |

| Other finding | Result |
| --- | --- |
| Right elbow | Missing in fall frames 66–67. |
| Right eye and right ear | Never meet the threshold anywhere in this clip. |

Coordinate availability and confidence do not guarantee precise joint positions. These results support further testing; they are not a full accuracy evaluation.

## 11. Running the current scripts

Set the shared settings in `falldet/config.py` before running:

| Setting | Purpose |
| --- | --- |
| `DATASET_ROOT` | Location of the extracted FallDataset folder on your computer. |
| `CHECK_DATASET` | Subset to inspect with `scripts/data/check_dataset.py`. |
| `POSE_VIDEO_ID` | Indexed video to process, such as `Home_01/video (1)`. |
| `POSE_OUTPUT_NAME` | Folder name for the pose-test results inside `outputs/`. |

Run the scripts from the project directory with your Python environment active.

| Order | Script | When to run |
| ---: | --- | --- |
| 1 | `scripts/data/clean_annotations.py` | If the cleaned Home annotation folders do not exist. Preserves the originals. |
| 2 | `scripts/data/check_dataset.py` | Check the selected subset without modifying files. |
| 3 | `scripts/data/build_dataset_index.py` | Build the index using original Coffee annotations and cleaned Home annotations. |
| 4 | `scripts/pose/extract_pose.py` | Extract poses from the video selected in `falldet/config.py`. |

If the cleaned Home folders already exist, start at Step 2.

```bash
python -m scripts.data.clean_annotations
python -m scripts.data.check_dataset
python -m scripts.data.build_dataset_index
python -m scripts.pose.extract_pose
```

To check another subset, change `CHECK_DATASET` in `falldet/config.py` and rerun `scripts/data/check_dataset.py`.

Cleanup runs separately in `scripts/data/clean_annotations.py`; there is no Step 16 to uncomment in the current check script. The single-video extraction script processes one selected video per run. Batch extraction is available in `scripts/pose/extract_pose_batch.py`; 129 batch pose files have now been processed by the window builder. See Section 16 for the window-building commands.

For installation instructions, see [Portable setup](docs/PORTABLE_SETUP.md).

## 12. Interpreting console messages

| Message | Meaning |
| --- | --- |
| `READ OK` | Decoded frames match the reported video frame count. |
| `COUNT OK` | Annotation row count matches the video frame count. |
| `ORDER OK` | Frame IDs are complete and ordered. |
| `CHECK` or `CHECK FRAME ORDER` | A mismatch needs inspection. |
| `EXACT DUPLICATES ONLY` | Removing identical complete rows restores the expected frame sequence. |
| `NEEDS MORE REVIEW` | Exact duplicate removal does not fully resolve the sequence. |
| `SAVED` | A cleaned annotation was written after validation. |
| `NOT SAVED — REVIEW` | The cleaning step did not save an unresolved annotation. |
| `[mp3float ...] Header missing` | Audio decoding message observed for some AVI files. |
| `Process finished with exit code 0` | The program ended without an unhandled exception. |

The audio messages did not prevent the successful video checks and output saves reported here. Video readability was verified separately. Printed check warnings still matter even when the exit code is zero.

## 13. Unresolved items and next steps

| Item | Next action |
| --- | --- |
| Window dataset | Built and checked; complete documentation and group code review before closing issue #6. |
| Remaining missing values | Keep NaN coordinates; verify downstream feature/classifier handling of NaN features. |
| Zero-zero review | Completed for all 31 included zero-zero videos; retain review records. |
| Fall/post-fall labels | Current implemented rules and comparison results are in Section 16; confirm them in group review. |
| Coffee_room_01/video (35) | Resolve conflicting annotations or retain the exclusion. |
| Office and Lecture_room | Find annotations or label them before supervised inclusion. |
| Data split | Keep windows from a source video in one split; consider subject/scene separation when metadata permits. |
| Classifier | Build features, train, and evaluate after data preparation. |
| Deployment | Test the complete system on Jetson Orin Nano later. |


## 14. Home_01 Pose Test.

We tested the pretrained YOLO11n-Pose model on `Home_01/video (1)`
to check pose extraction in a different room and camera angle
from the earlier Coffee_room_01 test.

### What the script did

1. Found the video in `data/dataset_index.csv`.
2. Resolved its relative path using the dataset root in `falldet/config.py`.
3. Processed every frame and detected body keypoints.
4. Saved a preview video showing the detected skeletons.
5. Saved the numerical keypoints in `keypoints.npz`.

### Results

| Check | Result |
|---|---|
| Frames processed | 264 |
| Frames with one person detected | 261 |
| Frames with no person detected | 0 |
| Frames with multiple detections | 3 |
| Multiple-detection frame numbers | 124, 125, 127 |
| Keypoints array shape | `(264, 17, 3)` |
| Annotated fall interval | Frames 144–164, inclusive |
| Detections during the fall | One person in all 21 frames |

The array contains 264 frames, 17 body keypoints per frame,
and three values per keypoint: normalized x, normalized y,
and confidence.

### Handling uncertain detections

Frames with multiple detections keep their position in the array,
but their keypoints are marked as missing (`NaN`).

For frames with one detection, coordinates of keypoints with
confidence below 0.5 are marked as missing. Their confidence
values are preserved.

### Review and conclusion

Sampled preview frames showed that the skeleton generally followed
the person while standing, falling, and lying down.

During the annotated fall interval, both shoulders, hips, knees,
and ankles had confidence of at least 0.5 in all 21 frames.
Some face and arm keypoints were less reliable.

This confirms successful pose extraction on this Home video.
It does not measure fall-classifier accuracy; no Fall/Normal
classifier was trained or evaluated in this test.

The test was repeated successfully after introducing the shared
configuration and relative CSV paths, with the same frame and
detection counts.

### Saved outputs

Local output folder:

`outputs/pose_check_home01_video01/`

It contains the preview video and `keypoints.npz`.
Generated outputs are excluded from Git.

For installation, dataset configuration, and run instructions,
see [Portable setup](docs/PORTABLE_SETUP.md).

## 15. Privacy by design

The system is designed for nursing-home rooms, where residents should not be recorded. It keeps as little visual data as possible.

| Principle | What it means |
| --- | --- |
| On-device processing only | Pose estimation and fall detection run locally on the device (target: Jetson Orin Nano). No cloud service or remote server processes the video. |
| Frames deleted after pose extraction | Each camera frame is kept in memory only while the pose model processes it, then discarded. Frames are not written to disk. |
| Only keypoints are stored | The only saved data are body keypoints (17 joint coordinates and confidences per frame) and the detection results derived from them. |
| No video sent anywhere | No video or images leave the device. Network access is only needed once, to download the pretrained model weights. |

During development, the scripts in this repository also read the public FallDataset videos and can save preview videos for checking results. These development outputs stay on the local computer and are excluded from Git; they are not part of the deployed system.

## 16. Issue #6 — Labelled pose windows

`build_windows.py` reads the per-video poses in `outputs/poses/`, assigns window labels, interpolates short gaps, filters poor-quality windows, and saves `outputs/windows.npz`. Original pose files remain unchanged. The following are the implemented rules for the verified build on 9 October 2026; merging and final group review remain separate steps.

### Window timing and labels

| Setting | Implemented rule |
| --- | --- |
| Window length | 30 consecutive frames; complete windows only, without padding |
| Step size | 5 frames; window starts are 1, 6, 11, and so on |
| Coffee room timing | 25 FPS: 1.2-second windows and 0.2-second steps |
| Home timing | 24 FPS: 1.25-second windows and approximately 0.208-second steps |
| Normal in reviewed no-fall videos | Every complete candidate window in the 31 reviewed `0,0` videos, subject to quality filtering |
| Normal in fall videos | Window ends strictly before `fall_start` |
| Fall | Overlap is at least `0.5 × min(30, fall_length)` frames |
| Entirely post-fall | Exclude windows starting after `fall_end` |
| Fall and landing | Keep qualifying Fall windows even if they end after `fall_end`; no extra one-second limit |
| Ambiguous partial overlap | Exclude windows overlapping the fall that do not meet the Fall rule; do not label them Normal |

Annotation frame numbers start at 1 and intervals are inclusive:

```python
fall_length = fall_end - fall_start + 1
overlap = max(
    0,
    min(window_end, fall_end) - max(window_start, fall_start) + 1,
)
required_overlap = 0.5 * min(30, fall_length)
```

The overlap rule is adaptive: a 14-frame fall requires at least 7 overlapping frames, while a fall lasting 30 frames or more requires at least 15. It is not always a requirement that half of the full window contains falling.

These windows retain native frame timing; the videos were not resampled to a common FPS. Thirty frames are therefore not exactly one second, or exactly 1.2 seconds, in every subset.

### Why omit the extra one-second post-fall limit?

Both comparisons used 30-frame windows, step 5, verified per-dataset FPS, and the flexible quality filter below.

| Post-fall option | Fall candidates | Retained Fall windows | Removed by quality | Falls represented |
| --- | ---: | ---: | ---: | ---: |
| Exclude windows starting after `fall_end` | 630 | 576 | 54 (8.6%) | 96/98 |
| Same rule, plus end within one second after `fall_end` | 628 | 574 | 54 (8.6%) | 96/98 |

The extra limit removes only two otherwise retained windows and does not change fall coverage. Following the group suggestion to keep the rule simple when counts are similar, the implemented builder omits that extra limit. This comparison does not establish which option gives better classifier accuracy or fewer false alarms.

### Short-gap interpolation and quality filtering

| Item | Behaviour |
| --- | --- |
| Interpolation | Linear interpolation of x and y for each of the 17 joints separately |
| Maximum gap | `floor(FPS × 0.12)` consecutive missing observations: 3 frames at 25 FPS, 2 at 24 FPS |
| Endpoints | Valid joint coordinates must exist on both sides inside the same window |
| Long or boundary gaps | Left missing; no extrapolation |
| Original input | Unchanged; processing uses a copy |
| Confidence | Original confidence values are preserved, including at interpolated positions |
| Remaining missing coordinates | Stored as NaN, not zero; both coordinates of a remaining missing joint observation are set to NaN |
| Core joints | Left/right shoulders and hips: indices 5, 6, 11, 12 |
| Missing core observation | Either x or y is not finite |
| Quality threshold | Exclude only when more than 30% of core-joint observations remain missing after interpolation |

Quality is measured over **30 frames × 4 joints = 120 observations**. At most 36 may remain missing. This is not the percentage of entirely missing frames and does not require every non-core joint to be present.

The gap duration convention is missing-run length divided by FPS. Filled values are estimates, not new detections; `interpolated_mask` records them. Low original confidence does not get increased when coordinates are filled.

`falldet/features.py` computes a different `missing_frac` feature: the fraction of frames with a missing hip or shoulder midpoint y-coordinate. That feature is not the builder's core-observation quality fraction. Feature aggregation ignores NaNs where possible, but an all-missing feature segment can still produce a NaN feature; downstream classifier handling must be checked explicitly.

### Run the checks and builder

From the project root, with the project's Python environment active:

```bash
python test_actual_interpolation.py
python build_windows.py
```

The input files must already exist in `outputs/poses/`. Paths are obtained from `falldet/config.py`; the builder's window, FPS, and filtering settings are currently defined in `build_windows.py`. Running the builder creates or replaces `outputs/windows.npz`.

The input coordinates must already be normalized by image width and height. The builder preserves their scale and does not convert pixel coordinates to normalized coordinates.

### Saved arrays

Here, `n` is the number of retained windows, currently 6,106.

| Field | Shape | Meaning |
| --- | --- | --- |
| `X` | `(n, 30, 17, 3)` | Float32 keypoints: normalized x, normalized y, confidence; NaNs may remain |
| `y` | `(n,)` | Integer labels: Normal = 0, Fall = 1 |
| `video_id` | `(n,)` | Subset-qualified source video IDs |
| `window_start`, `window_end` | `(n,)` each | Original one-based inclusive frame boundaries |
| `fps` | `(n,)` | Reported source FPS used for each window |
| `core_missing_fraction` | `(n,)` | Remaining missing core-observation fraction after interpolation |
| `interpolated_mask` | `(n, 30, 17)` | True where joint coordinates were estimated |

The file also records `window_length`, `step_size`, `max_gap_seconds`, `max_missing_fraction`, `core_joints`, `fall_overlap_rule`, and `postfall_rule`. Dropped windows are not included in these arrays.

### Verified build results

| Category | Candidates before quality filtering | Retained | Removed by quality |
| --- | ---: | ---: | ---: |
| Normal from reviewed `0,0` videos | 2,380 | 2,306 | 74 |
| Normal before falls | 3,294 | 3,224 | 70 |
| **All Normal** | **5,674** | **5,530** | **144** |
| **Fall** | **630** | **576** | **54** |
| **Combined** | **6,304** | **6,106** | **198** |

| Additional check | Result |
| --- | --- |
| Pose files processed | 129 |
| All complete windows considered | 7,675 |
| Windows excluded by labelling rules | 1,371 |
| Falls with at least one retained Fall window | 96/98 |
| Fall videos contributing pre-fall Normal windows | 97/98 |
| Saved `X` shape | `(6106, 30, 17, 3)` |
| Saved `y` and `video_id` shapes | `(6106,)` each |
| Duplicate video/start-frame pairs | None |
| Maximum remaining core missing fraction | 0.30; no saved window exceeds the limit |
| Finite x/y coordinate range in the saved file | 0 to 1 |
| Infinite x/y coordinates | None; missing coordinates remain NaN |
| Saved-file verification | Passed |

`Home_01/video (11)` and `Home_02/video (35)` have no retained Fall windows because all their qualifying Fall windows fail the quality rule. Their valid pre-fall Normal windows are still included. `Coffee_room_01/video (17)` has too few frames before the fall to form a complete 30-frame Normal window; this is not a quality-filter rejection.

### Actual interpolation verification

`test_actual_interpolation.py` checked all 7,675 complete windows from the 129 pose files. It verified that actual remaining-missing masks match the estimates, filled coordinates follow linear interpolation, and original inputs and confidence values remain unchanged. There were 38,577 interpolated joint observations across overlapping windows; this includes repeated observations across windows and is not a count of unique source-video joints.

The check passed again after updating imports to the reorganized `falldet` package. It verifies implementation behaviour, not whether interpolated coordinates exactly match the true body position.

### Training, deployment, and issue completion

- Keep overlapping windows from a source video in the same train/validation/test split. Consider subject or scene grouping where metadata permits.
- Normal windows substantially outnumber Fall windows. Report class-specific evaluation; these counts alone are not model accuracy.
- Remaining NaNs must be supported by the feature/classifier pipeline; do not silently replace missing body coordinates with zero.
- The Jetson implementation must match coordinate normalization, window timing, interpolation, and quality filtering. Reuse shared preprocessing where possible. Native 24/25-FPS windows do not by themselves solve variable live-camera or processing rates.
- Fall annotations are used only for offline labels; live inference does not have `fall_start` or `fall_end` available.
- Live preprocessing parity, classifier evaluation, and deployment are not verified by the dataset-building checks.
- Generated `outputs/` files are excluded from Git. Commit the builder, preprocessing, tests, and documentation so teammates can reproduce the file.
- Complete group review and merge the issue #6 changes before marking the issue closed.

## 17. References and evidence

The numerical results here come from the project's dataset checks, manual video reviews, pose tests, window-rule comparisons, and saved-file verification. The supplied dataset README was used during initial inspection. Reported FPS was subsequently checked for all 129 indexed videos: 69 Coffee videos report 25 FPS and 60 Home videos report 24 FPS. These preparation checks do not measure fall-classifier accuracy.

Sections 9, 10, and 14 describe the original single-video experiments; their detection settings should not be assumed to describe every later batch-extraction run. Inspect the metadata saved in each pose file for its extraction settings.
