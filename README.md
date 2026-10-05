# Fall Detection: Dataset Preparation and Pose Extraction

  
**Current stage:** Dataset checks and annotation cleanup completed for the included data; first pose-extraction test completed.

This README records the work completed from the first dataset check to the first YOLO11n Pose test. It explains what each script does, which files were changed, what the results mean, and what remains to be done.

## 1. Project overview

The planned system uses camera video, YOLO11n Pose body keypoints, movement features across frames, and a lightweight Fall/Normal classifier. The intended deployment device is a Jetson Orin Nano.

| Stage | Current status |
| --- | --- |
| Locate dataset folders | Completed |
| Validate the four annotated subsets | Completed for the checks described below |
| Clean exact duplicate Home annotation rows | Completed; originals preserved |
| Create the dataset index | Completed: 129 videos |
| Inspect two videos with `0, 0` fall values | Completed using sampled frames |
| Extract and inspect Coffee_room_01/video (1) poses | Completed |
| Test pose extraction on Home_01/video (1) | Completed |
| Extract poses for all included videos | Pending |
| Define final training labels and missing-data handling | Pending |
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
| Project directory | `C:\Users\haifa\Desktop\fall_detection` |
| Dataset root | `C:\Users\haifa\Desktop\FallDataset` |
| Libraries used | OpenCV, NumPy, Ultralytics, and Python's standard library |



### Scripts and outputs

| File or folder | Purpose |
| --- | --- |
| `check_dataset.py` | Read-only checks of folders, paired files, video decoding, annotation counts, frame order, and duplicates. Does not modify files. |
| `clean_annotations.py` | Removes exact duplicate annotation rows and saves validated copies in separate cleaned folders, preserving the originals. |
| `config.py` | Stores shared settings, including the dataset root, selected subset, pose-test video, and output paths. |
| `build_dataset_index.py` | Validate selected annotations and create one CSV row per included video. |
| `extract_pose.py` | Run pose inference on one indexed video and save numerical keypoints and a preview. |
| `dataset_index.csv` | Index of 129 included videos and their selected annotations. |
| `Annotation_files_cleaned` | Separate cleaned annotation folder inside each Home subset. |
| `outputs/pose_check_coffee01_video01` | Output folder for the completed Coffee pose test. |
| `keypoints.npz` | Compressed NumPy file containing frame-aligned keypoints and metadata. |
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

`check_dataset.py` was built incrementally and run separately for each annotated subset. Steps 11–14 search the wider dataset root; the other steps use the selected subset.

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
| 16 | Cleanup now runs separately in `clean_annotations.py`. | Saves validated cleaned copies while preserving originals. |

Steps 1–16 describe how the checking workflow developed. In the current version, `check_dataset.py` performs read-only checks.

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

These examples support the working interpretation **“no annotated fall.”** They do not verify every zero-zero clip or establish an official dataset-wide convention. We preserved the original values and used the separate index status `zero_zero`.

## 8. Building the dataset index

`build_dataset_index.py` uses original Coffee annotations, cleaned Home annotations, and the scoped exclusion of Coffee_room_01/video (35).

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

The index script reads video metadata. Full-frame decoding was already performed in Step 6 of `check_dataset.py`.

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
| `visual_review` | `sampled_frames_no_fall_seen` for the two reviewed zero-zero examples; otherwise `not_recorded_in_this_index`. |

The review column is limited: earlier inspection of Coffee_room_01/video (26) is documented here but was not recorded in that field. Subset-qualified IDs prevent confusion between videos with the same filename.

The CSV is saved beside the script as `dataset_index.csv`. Rebuilding it overwrites that output. Video and annotation paths are relative to `DATASET_ROOT`, which is configured in `config.py`. Teammates can change the dataset root without rebuilding the CSV, provided the dataset's internal folder structure stays the same..

## 9. First pose-extraction test

`extract_pose.py` was tested on **Coffee_room_01/video (1)**. It uses a pretrained model for inference; no classifier or pose model was trained during this test.

### Extraction process

| Step | Action |
| ---: | --- |
| 1 | Find the selected video in `dataset_index.csv`. |
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

Set the shared settings in `config.py` before running:

| Setting | Purpose |
| --- | --- |
| `DATASET_ROOT` | Location of the extracted FallDataset folder on your computer. |
| `CHECK_DATASET` | Subset to inspect with `check_dataset.py`. |
| `POSE_VIDEO_ID` | Indexed video to process, such as `Home_01/video (1)`. |
| `POSE_OUTPUT_NAME` | Folder name for the pose-test results inside `outputs/`. |

Run the scripts from the project directory with your Python environment active.

| Order | Script | When to run |
| ---: | --- | --- |
| 1 | `clean_annotations.py` | If the cleaned Home annotation folders do not exist. Preserves the originals. |
| 2 | `check_dataset.py` | Check the selected subset without modifying files. |
| 3 | `build_dataset_index.py` | Build the index using original Coffee annotations and cleaned Home annotations. |
| 4 | `extract_pose.py` | Extract poses from the video selected in `config.py`. |

If the cleaned Home folders already exist, start at Step 2.

```bash
python clean_annotations.py
python check_dataset.py
python build_dataset_index.py
python extract_pose.py
```

To check another subset, change `CHECK_DATASET` in `config.py` and rerun `check_dataset.py`.

Cleanup runs separately in `clean_annotations.py`; there is no Step 16 to uncomment in the current check script. Pose extraction currently processes one selected video per run.

For installation instructions, see [Portable setup](PORTABLE_SETUP.md).

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
| Remaining indexed videos | Extract poses and record per-video quality results. |
| Missing joints and multiple detections | Define handling that preserves frame alignment. |
| Zero-zero metadata | Complete the interpretation and review required before assigning final training labels. |
| Fall/post-fall labels | Define how the interval maps to frames and temporal windows; do not automatically treat all post-fall lying frames as Normal. |
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

1. Found the video in `dataset_index.csv`.
2. Resolved its relative path using the dataset root in `config.py`.
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
see [Portable setup](PORTABLE_SETUP.md).

## 15. References and evidence


