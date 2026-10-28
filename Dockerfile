# Fall detection on Jetson Nano (JetPack 4.6.x).
# Base image = Ultralytics' Jetson image: Python 3.8 + GPU PyTorch + YOLO already inside.
# (The Jetson itself only has Python 3.6, which is too old for ultralytics.)
FROM ultralytics/ultralytics:latest-jetson-jetpack4

# Our extra packages. scikit-learn MUST be the same version that saved
# models/fall_rf.joblib on the laptop (check: pip show scikit-learn).
ARG SKLEARN_VERSION=1.3.2
RUN pip install --no-cache-dir "scikit-learn==${SKLEARN_VERSION}" joblib paho-mqtt

# Our code + trained model + YOLO weights
WORKDIR /app
COPY . /app

# Defaults for running in a room corner without a screen
ENV HEADLESS=1 \
    CAMERA=tcp://127.0.0.1:5000

CMD ["python3", "live_detect.py"]