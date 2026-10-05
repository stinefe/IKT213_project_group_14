# Portable setup for the fall detection project

This update centralizes paths and stores dataset-relative paths in the index.
It preserves the current YOLO inference settings and missing-keypoint policy.

## Files

| File | Purpose |
|---|---|
| `config.py` | Dataset root, project paths, subset and pose-test selection |
| `check_dataset.py` | Existing validation steps 1–15; now read-only |
| `clean_annotations.py` | Recreate cleaned annotations for both Home subsets from originals |
| `build_dataset_index.py` | Validate and index the same 129 videos using relative paths |
| `extract_pose.py` | Resolve relative CSV paths using the configured dataset root |
| `requirements.txt` | Direct dependencies; versions not yet pinned |
| `gitignore_additions.txt` | Rules to append to your existing `.gitignore` |

## Install this update

1. Keep a backup of your current scripts. Copy the Python files and
   `requirements.txt` into your project beside the existing `README.md`.
2. Append the contents of `gitignore_additions.txt` to the existing `.gitignore`.
   The additions file is not itself an active Git ignore file.
3. Open `config.py`. The default layout is a project folder and a `FallDataset`
   folder beside each other. This matches Haifaa's Desktop layout. If needed,
   change only `DATASET_ROOT` to your own dataset location, using `Path(...)`.
   Each teammate configures that line locally; do not replace teammates' paths.
4. Use your project Python environment. Install dependencies if needed:

   ```console
   python -m pip install -r requirements.txt
   ```

   No dependency upgrade is required just to test this update in the existing
   working `ikt213` environment. The model may need internet on its first run;
   an existing `yolo11n-pose.pt` beside the scripts will be reused.

## Run order

Run these commands from the project directory (or run each file in PyCharm):

| Order | Command | Expected result |
|---|---|---|
| 1 | `python clean_annotations.py` | Both Home cleaned folders created or unchanged |
| 2 | `python check_dataset.py` | Default Home_02: 30 READ OK, COUNT OK and ORDER OK |
| 3 | `python build_dataset_index.py` | 129 indexed, 98 fall intervals, 31 zero-zero |
| 4 | `python extract_pose.py` | Home_01/video (1), 264 frames and `(264, 17, 3)` keypoints |

Haifaa already has both cleaned folders. She can start at step 2.
New teammates must run step 1 if they only have original annotations.
The cleaner reads originals every time, preserves the single-value metadata,
removes only exact six-column duplicates, and refuses unresolved frame-order
or interval problems. It validates all 60 files before writing. Original
annotation files are never overwritten.

Choose each other labeled subset through `CHECK_DATASET` in `config.py` to
repeat validation. Coffee_room_01/video (35) remains a known count failure;
its exclusion from the index is intentional. No Office or Lecture_room labels
are invented. Those folders remain outside the supervised index.

**Rebuild the CSV before running the updated extraction script.** The previous
CSV contains absolute Windows paths and will be rejected. New paths look like:

```text
Home_01/Home_01/Videos/video (1).avi
Home_01/Home_01/Annotation_files_cleaned/video (1).txt
```

Both paths are relative to `DATASET_ROOT`, not to the current working directory.
The new CSV can be committed and shared. Keep subset names in `video_id` because
video filenames repeat between rooms.

 ## Home_01 pose test — verified before and after the portability update

| Check | Observed result |
|---|---|
| Video | Home_01/video (1) |
| Frames / keypoints shape | 264 / `(264, 17, 3)` |
| One / zero / multiple detections | 261 / 0 / 3 |
| Multiple-detection frames | 124, 125, 127 |
| Annotated fall interval | 144–164 inclusive |
| Detection during fall | One detection in all 21 frames |
| Shoulders, hips, knees, ankles during fall | Confidence at least 0.5 in all 21 frames |
| Visual review | Sampled preview frames generally follow standing, falling and lying |

Three ambiguous frames retain missing keypoints rather than being dropped.
Some face and arm points are uncertain. This test is evidence that pose
extraction works on this clip, not a fall-classifier accuracy measurement.
A rerun with different dependency versions can produce different detections.

## Before merging into main

- Local verification completed: the updated index builder saved 129 videos
  (98 fall intervals and 31 zero-zero entries). The updated pose script
  successfully read the relative video path and processed 264 frames,
  producing keypoints with shape `(264, 17, 3)`.
  Detection counts remained 261 single-person, 0 no-person, and 3 multiple.
- Record tested package versions from the working interpreter before pinning:

  ```console
  python -c "import sys; from importlib.metadata import version; print(sys.version); [print(p + '==' + version(p)) for p in ('numpy', 'opencv-python', 'ultralytics', 'torch', 'torchvision')]"
  ```

  Do not guess version numbers. The supplied requirements file lists direct
  dependencies only. Jetson deployment dependencies need separate validation.
- Review `pose_test.py` too: it was visible on GitHub but was not supplied for
  this update. Any personal paths there should also use the shared config.
- Update the existing README with this setup and the Home test results.
- Review the files included in the commit. Ignore rules do not untrack files
  that Git already tracks. Dataset videos, generated outputs and model weights
  should not be committed.
- Push the verified update to Haifaa's branch, review the existing pull request,
  and merge into `main` when its changes and checks are ready. No merge has been
  performed by this update.
