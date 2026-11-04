# Fall detection on Jetson Nano (JetPack 4.6.x).
# Base image = Ultralytics' Jetson image: Python 3.8 + GPU PyTorch + YOLO already inside.
# (The Jetson itself only has Python 3.6, which is too old for ultralytics.)
FROM ultralytics/ultralytics:latest-jetson-jetpack4

# Our extra packages. scikit-learn MUST be the same version that saved
# models/fall_rf.joblib on the laptop (check: pip show scikit-learn).
ARG SKLEARN_VERSION=1.3.2
RUN pip install --no-cache-dir "scikit-learn==${SKLEARN_VERSION}" joblib paho-mqtt

# GUI-enabled OpenCV so cv2.imshow works when HEADLESS=0.
# Same version as the headless one in the base image; --no-deps keeps numpy untouched.
RUN apt-get update && apt-get install -y --no-install-recommends libgtk2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip uninstall -y opencv-python-headless \
    && pip install --no-cache-dir --no-deps "opencv-python==5.0.0.93"

# Our code + trained model + YOLO weights
WORKDIR /app
COPY . /app

# Defaults for running in a room corner without a screen
ENV HEADLESS=1 \
    CAMERA=tcp://127.0.0.1:5000

CMD ["python3", "live_detect.py"]