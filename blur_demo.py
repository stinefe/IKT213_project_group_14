"""Live, face-blurred pose preview from a camera or a video file. Nothing is saved.

Examples (run from the project folder):
    python blur_demo.py                    # default camera (index 0)
    python blur_demo.py --camera 1         # another camera
    python blur_demo.py "Home_01/Home_01/Videos/video (1).avi"   # relative to DATASET_ROOT (Video path can be copied from dataset_index.csv)
    python blur_demo.py "<video>" --skeleton --show-regions

Defaults: only the face blur is shown (no skeleton). Video files are shown 3x larger;
camera input keeps its own size, since a 3x webcam frame would not fit on the screen.
Use --scale to override either.

Keys: q or Esc quits, space pauses.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import DATASET_ROOT, MODEL_PATH
from face_blur import blur_regions, regions_from_result

VIDEO_SCALE = 3.0    # default enlargement for video files (dataset videos are 320x240)
CAMERA_SCALE = 1.0   # default for camera input

# Outline colours for --show-regions (BGR)
REGION_COLORS = {"face": (0, 255, 0), "fallback": (0, 165, 255)}


def find_video(value):
    """Return the video path, given as is or relative to DATASET_ROOT."""
    candidates = [Path(value), DATASET_ROOT / value]
    for path in candidates:
        if path.is_file():
            return path
    tried = "\n  ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Video not found: {value}\nLooked at:\n  {tried}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", nargs="?",
                        help="video file path; leave out to use the camera")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index when no video is given (default: 0)")
    parser.add_argument("--scale", type=float, default=None,
                        help=f"window size factor (default: {VIDEO_SCALE:g} for videos, "
                             f"{CAMERA_SCALE:g} for the camera)")
    parser.add_argument("--skeleton", action="store_true",
                        help="also draw person boxes and keypoints")
    parser.add_argument("--show-regions", action="store_true",
                        help="outline blurred areas: green = face keypoints, orange = fallback")
    parser.add_argument("--device", default="cpu", help="inference device (default: cpu)")
    args = parser.parse_args()

    if args.video:
        video = find_video(args.video)
        cap, name = cv2.VideoCapture(str(video)), video.name
        scale = VIDEO_SCALE if args.scale is None else args.scale
    else:
        cap, name = cv2.VideoCapture(args.camera), f"camera {args.camera}"
        scale = CAMERA_SCALE if args.scale is None else args.scale

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {name}")

    model = YOLO(str(MODEL_PATH))
    window = f"Face blur: {name}"
    frames = 0
    region_counts = {"face": 0, "fallback": 0}
    paused = False

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                # Inference runs on the original frame; only the displayed copy is blurred.
                result = model.predict(
                    frame, device=args.device, imgsz=640, conf=0.25, verbose=False
                )[0]
                regions = regions_from_result(result)
                shown = blur_regions(frame, regions)

                if args.skeleton:
                    shown = result.plot(img=shown)  # draw on the blurred copy

                if args.show_regions:
                    for (x1, y1, x2, y2), kind in regions:
                        cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)),
                                      REGION_COLORS[kind], 1)

                frames += 1
                for _, kind in regions:
                    region_counts[kind] += 1

                if scale != 1:
                    shown = cv2.resize(shown, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_LINEAR)
                cv2.imshow(window, shown)

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the mouse
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Frames shown:", frames)
    print("Blurs from face keypoints:", region_counts["face"])
    print("Fallback blurs:", region_counts["fallback"])


if __name__ == "__main__":
    main()