from pathlib import Path
from ultralytics import YOLO

# Find the folder containing this Python file.
folder = Path(__file__).resolve().parent

# Load the pretrained pose model.
model = YOLO("yolo11n-pose.pt")

# Process the video and display the skeleton.
results = model.predict(
    source=str(folder / "test_video.mp4"),
    device="cpu",
    iou=0.5,
    show=True,
    save=True,
    stream=True,
    project=str(folder / "outputs"),
    name="pose_test"
)

# Process each frame in the video.
for result in results:
    pass