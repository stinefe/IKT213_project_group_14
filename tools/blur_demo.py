"""Live, face-blurred pose preview from a camera or a video file. Nothing is saved.

Examples (run from the project folder):
    python blur_demo.py                    # default camera (index 0)
    python blur_demo.py --camera 1         # another camera
    python blur_demo.py "Home_01/Home_01/Videos/video (1).avi"   # relative to DATASET_ROOT (Video path can be copied from dataset_index.csv)
    python blur_demo.py "<video>" --skeleton --show-regions
    python3 blur_demo.py --csi --device 0  # Jetson Nano with a CSI ribbon camera
    python blur_demo.py udp://127.0.0.1:5000   # network stream (anything with "://")
    python3 blur_demo.py --csi --backend trt   # Jetson Nano, TensorRT engine instead of Ultralytics

Defaults: only the face blur is shown (no skeleton). Video files are shown 3x larger;
camera input keeps its own size, since a 3x webcam frame would not fit on the screen.
Use --scale to override either.

Keys: q or Esc quits, space pauses.
"""
import argparse
import time
from pathlib import Path

import cv2

from falldet.config import DATASET_ROOT, MODEL_PATH
from falldet.face_blur import blur_regions, face_regions, regions_from_result
# ultralytics and falldet.trt_pose are imported lazily in main(), one or the other,
# so the script runs on the Jetson without ultralytics/torch installed.

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


def csi_pipeline(sensor_id, width, height, fps, flip):
    """Build a GStreamer pipeline string for a Jetson CSI camera (nvarguscamerasrc)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def draw_skeleton(frame, boxes, keypoints_xy, keypoints_conf, conf_thres=0.5):
    """Draw person boxes and visible keypoints on frame, in place; returns frame.

    Used for --backend trt, where there is no Ultralytics result to call .plot() on.
    """
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
    for xy, conf in zip(keypoints_xy, keypoints_conf):
        for (x, y), c in zip(xy, conf):
            if c >= conf_thres:
                cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 255, 255), -1)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", nargs="?",
                        help="video file path; leave out to use the camera")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index when no video is given; with --csi, the sensor id "
                             "(0 = CAM0) (default: 0)")
    parser.add_argument("--csi", action="store_true",
                        help="use a Jetson CSI ribbon camera via GStreamer instead of a USB camera")
    parser.add_argument("--csi-width", type=int, default=1280, help="CSI capture width (default: 1280)")
    parser.add_argument("--csi-height", type=int, default=720, help="CSI capture height (default: 720)")
    parser.add_argument("--csi-fps", type=int, default=30, help="CSI capture framerate (default: 30)")
    parser.add_argument("--flip", type=int, default=0,
                        help="nvvidconv flip-method, e.g. 2 = rotate 180 deg (default: 0)")
    parser.add_argument("--scale", type=float, default=None,
                        help=f"window size factor (default: {VIDEO_SCALE:g} for videos, "
                             f"{CAMERA_SCALE:g} for the camera)")
    parser.add_argument("--skeleton", action="store_true",
                        help="also draw person boxes and keypoints")
    parser.add_argument("--show-regions", action="store_true",
                        help="outline blurred areas: green = face keypoints, orange = fallback")
    parser.add_argument("--device", default="cpu", help="inference device (default: cpu)")
    parser.add_argument("--backend", choices=("ultralytics", "trt"), default="ultralytics",
                        help="pose inference backend; trt needs a Jetson + TensorRT engine "
                             "(default: ultralytics)")
    parser.add_argument("--engine", default="models/yolo11n-pose.engine",
                        help="TensorRT engine path, used with --backend trt "
                             "(default: models/yolo11n-pose.engine)")
    parser.add_argument("--profile", action="store_true",
                        help="also print average per-stage time in ms, every ~2 seconds")
    args = parser.parse_args()

    if args.video and "://" in args.video:
        cap, name = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG), args.video
        scale = CAMERA_SCALE if args.scale is None else args.scale
    elif args.video:
        video = find_video(args.video)
        cap, name = cv2.VideoCapture(str(video)), video.name
        scale = VIDEO_SCALE if args.scale is None else args.scale
    elif args.csi:
        pipeline = csi_pipeline(args.camera, args.csi_width, args.csi_height,
                                 args.csi_fps, args.flip)
        cap, name = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER), f"CSI camera {args.camera}"
        scale = CAMERA_SCALE if args.scale is None else args.scale
    else:
        cap, name = cv2.VideoCapture(args.camera), f"camera {args.camera}"
        scale = CAMERA_SCALE if args.scale is None else args.scale

    if not cap.isOpened():
        if args.csi:
            raise RuntimeError(
                f"Cannot open {name}. Common causes:\n"
                "  - OpenCV was built without GStreamer support "
                "(check cv2.getBuildInformation() for 'GStreamer: YES')\n"
                "  - The camera service is stuck: sudo systemctl restart nvargus-daemon"
            )
        raise RuntimeError(f"Cannot open {name}")

    if args.backend == "trt":
        from falldet.trt_pose import TrtPose
        engine = TrtPose(args.engine)
    else:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))

    window = f"Face blur: {name}"
    frames = 0
    region_counts = {"face": 0, "fallback": 0}
    paused = False
    fps_frames, fps_start = 0, time.time()
    # --profile: summed per-stage seconds since the last print, divided by fps_frames.
    stage_totals = {"capture": 0.0, "inference": 0.0, "regions": 0.0,
                     "blur": 0.0, "display": 0.0}

    try:
        while True:
            if not paused:
                t0 = time.time()
                ok, frame = cap.read()
                t1 = time.time()
                if not ok:
                    break

                # Inference runs on the original frame; only the displayed copy is blurred.
                if args.backend == "trt":
                    boxes, keypoints_xy, keypoints_conf = engine(frame)
                    t2 = time.time()
                    regions = face_regions(boxes, keypoints_xy, keypoints_conf)
                else:
                    result = model.predict(
                        frame, device=args.device, imgsz=640, conf=0.25, verbose=False
                    )[0]
                    t2 = time.time()
                    regions = regions_from_result(result)
                t3 = time.time()

                shown = blur_regions(frame, regions)
                t4 = time.time()

                if args.skeleton:
                    if args.backend == "trt":
                        shown = draw_skeleton(shown, boxes, keypoints_xy, keypoints_conf)
                    else:
                        shown = result.plot(img=shown)  # draw on the blurred copy

                if args.show_regions:
                    for (x1, y1, x2, y2), kind in regions:
                        cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)),
                                      REGION_COLORS[kind], 1)

                frames += 1
                fps_frames += 1
                for _, kind in regions:
                    region_counts[kind] += 1

                if scale != 1:
                    shown = cv2.resize(shown, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_LINEAR)
                cv2.imshow(window, shown)
                t5 = time.time()

                if args.profile:
                    stage_totals["capture"] += t1 - t0
                    stage_totals["inference"] += t2 - t1
                    stage_totals["regions"] += t3 - t2
                    stage_totals["blur"] += t4 - t3
                    stage_totals["display"] += t5 - t4

                now = time.time()
                if now - fps_start >= 2.0:
                    fps = fps_frames / (now - fps_start)
                    if args.profile and fps_frames:
                        stages = "  ".join(
                            f"{stage} {1000 * total / fps_frames:.1f}ms"
                            for stage, total in stage_totals.items()
                        )
                        print(f"FPS: {fps:.1f}  {stages}")
                    else:
                        print(f"FPS: {fps:.1f}")
                    fps_frames, fps_start = 0, now
                    for stage in stage_totals:
                        stage_totals[stage] = 0.0

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            # -1 means the backend doesn't support this query (e.g. GTK on the Jetson's
            # system OpenCV 4.1.1); treat that as "still open" rather than "closed".
            visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
            if 0 <= visible < 1:
                break  # window closed with the mouse
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Frames shown:", frames)
    print("Blurs from face keypoints:", region_counts["face"])
    print("Fallback blurs:", region_counts["fallback"])


if __name__ == "__main__":
    main()