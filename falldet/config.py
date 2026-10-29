"""Shared project settings: paths, dataset options and MQTT alerts.
Change DATASET_ROOT here if your dataset is elsewhere."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default: FallDataset sits next to the project folder.
DATASET_ROOT = PROJECT_ROOT.parent / "FallDataset"
INDEX_FILE = PROJECT_ROOT / "data" / "dataset_index.csv"
SPLITS_FILE = PROJECT_ROOT / "data" / "splits.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = PROJECT_ROOT / "yolo11n-pose.pt"

ANNOTATION_FOLDERS = {
    "Coffee_room_01": "Annotation_files",
    "Coffee_room_02": "Annotations_files",
    "Home_01": "Annotation_files_cleaned",
    "Home_02": "Annotation_files_cleaned",
}

# Own annotations (fall start/end only) for subsets without original labels.
OWN_ANNOTATION_DATASETS = {
    "Lecture_room": "Lecture room",
    "Office": "Office",
}

CHECK_DATASET = "Home_02"
POSE_VIDEO_ID = "Home_01/video (1)"
POSE_OUTPUT_NAME = "pose_check_home01_video01"

MQTT_BROKER = "localhost"      # later: the Jetson's IP (or "localhost" on the Jetson itself)
MQTT_PORT = 1883
MQTT_TOPIC = "falldetection/alerts"          # full alert as JSON
MQTT_STATUS_TOPIC = "falldetection/status"   # simple "fall" / "ok" for LED indicators


def require_dataset_root():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}. Set DATASET_ROOT in config.py."
        )


def resolve_dataset_path(value):
    """Read only paths relative to DATASET_ROOT, as stored in the new CSV."""
    from pathlib import PureWindowsPath
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).drive or ".." in relative.parts:
        raise ValueError("Index has an absolute or invalid path. Run build_dataset_index.py again.")
    return DATASET_ROOT / relative
