"""Checks for the pure-numpy parts of falldet.trt_pose (letterbox, postprocess) and
for falldet.face_blur.face_regions accepting that output directly.

Runs on a PC with only numpy and cv2 -- no GPU, no tensorrt, no pycuda. TrtPose
itself (which needs those) is not touched here; see docs/jetson_tensorrt.md for
testing the full engine on the Jetson.
"""
import numpy as np

from falldet.face_blur import face_regions, regions_from_result
from falldet.trt_pose import COCO_KEYPOINT_COUNT, letterbox, postprocess


class FakeTensor:
    """Minimal stand-in for a torch tensor: .cpu().numpy() returns the array."""

    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = FakeTensor(xyxy)
        self._len = len(xyxy)

    def __len__(self):
        return self._len


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = FakeTensor(xy)
        self.conf = FakeTensor(conf)


class FakeResult:
    """Mimics the parts of an Ultralytics pose Result that regions_from_result reads."""

    def __init__(self, boxes_xyxy, keypoints_xy, keypoints_conf):
        self.boxes = FakeBoxes(boxes_xyxy)
        self.keypoints = FakeKeypoints(keypoints_xy, keypoints_conf)


def check_letterbox_shape_and_padding():
    """Output always has the requested size, and padding sits on one axis only."""
    size = (416, 416)

    for shape in [(240, 320, 3), (320, 240, 3), (416, 416, 3)]:  # landscape, portrait, square
        frame = np.zeros(shape, dtype=np.uint8)
        padded, scale, (left, top) = letterbox(frame, size)

        if padded.shape[:2] != size[::-1]:
            raise AssertionError(f"Wrong letterboxed shape for input {shape}: {padded.shape}")

        height, width = shape[:2]
        scaled_w, scaled_h = width * scale, height * scale
        if not (abs(scaled_w - size[0]) < 1.0 or abs(scaled_h - size[1]) < 1.0):
            raise AssertionError(f"Neither axis touches the target size for input {shape}")

        if left < 0 or top < 0 or left >= size[0] or top >= size[1]:
            raise AssertionError(f"Padding out of range for input {shape}: {(left, top)}")


def check_letterbox_roundtrip():
    """A point mapped into letterboxed space and back lands on the same original pixel,
    for both a landscape and a portrait frame."""
    size = (416, 416)

    for shape in [(240, 320, 3), (480, 270, 3)]:  # landscape, portrait
        frame = np.zeros(shape, dtype=np.uint8)
        _, scale, pad = letterbox(frame, size)
        height, width = shape[:2]

        points = np.array([[0, 0], [width - 1, height - 1], [width / 2, height / 3]],
                           dtype=np.float32)

        from falldet.trt_pose import scale_to_letterboxed, scale_to_original
        roundtrip = scale_to_original(scale_to_letterboxed(points, scale, pad), scale, pad)

        if not np.allclose(roundtrip, points, atol=1e-3):
            raise AssertionError(f"Round trip mismatch for shape {shape}: {roundtrip} vs {points}")


def _make_detection(cx, cy, w, h, box_conf, num_keypoints=COCO_KEYPOINT_COUNT,
                     keypoint_conf=0.9):
    """One row of length 5 + 3*num_keypoints, with all keypoints at the box centre."""
    row = [cx, cy, w, h, box_conf]
    for _ in range(num_keypoints):
        row += [cx, cy, keypoint_conf]
    return row


def _make_output(detections):
    """Stack rows into the engine's raw (1, 56, N) layout."""
    array = np.array(detections, dtype=np.float32)  # (N, 56)
    return array.T[None]  # (1, 56, N)


def check_postprocess_confidence_filter():
    """Detections below conf_thres are dropped."""
    size = (416, 416)
    frame_shape = (480, 640)
    _, scale, pad = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), size)

    detections = [
        _make_detection(100, 100, 40, 40, box_conf=0.9),
        _make_detection(300, 300, 40, 40, box_conf=0.1),  # below threshold
    ]
    output = _make_output(detections)

    boxes, keypoints_xy, keypoints_conf = postprocess(
        output, scale, pad, frame_shape, conf_thres=0.25, iou_thres=0.45
    )

    if len(boxes) != 1:
        raise AssertionError(f"Expected 1 detection above threshold, got {len(boxes)}")
    if boxes.shape != (1, 4) or keypoints_xy.shape != (1, COCO_KEYPOINT_COUNT, 2) \
            or keypoints_conf.shape != (1, COCO_KEYPOINT_COUNT):
        raise AssertionError("Wrong output shapes after confidence filtering")


def check_postprocess_nms():
    """Overlapping boxes collapse to one; separate people both survive."""
    size = (416, 416)
    frame_shape = (480, 640)
    _, scale, pad = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), size)

    detections = [
        _make_detection(100, 100, 40, 40, box_conf=0.90),  # person A, best box
        _make_detection(102, 101, 42, 39, box_conf=0.80),  # person A, overlapping duplicate
        _make_detection(103, 99, 38, 41, box_conf=0.70),   # person A, overlapping duplicate
        _make_detection(300, 300, 40, 40, box_conf=0.85),  # person B, far away
    ]
    output = _make_output(detections)

    boxes, _, _ = postprocess(output, scale, pad, frame_shape, conf_thres=0.25, iou_thres=0.45)

    if len(boxes) != 2:
        raise AssertionError(f"Expected 2 people after NMS, got {len(boxes)}")


def check_postprocess_scale_back_and_clip():
    """Boxes/keypoints land in original-frame pixels and are clipped to the frame."""
    size = (416, 416)
    frame_shape = (480, 640)  # height, width; letterboxing pads top/bottom (width is wider)
    _, scale, pad = letterbox(np.zeros(frame_shape + (3,), dtype=np.uint8), size)

    # Centred at the letterboxed image's own centre and edge, in letterboxed pixels.
    # Distinct confidences so sort order (and therefore boxes[0]) is unambiguous.
    centre_detection = _make_detection(size[0] / 2, size[1] / 2, 20, 20, box_conf=0.95)
    edge_detection = _make_detection(size[0] - 1, size[1] - 1, 40, 40, box_conf=0.85)
    output = _make_output([centre_detection, edge_detection])

    boxes, keypoints_xy, keypoints_conf = postprocess(
        output, scale, pad, frame_shape, conf_thres=0.25, iou_thres=0.45
    )

    if boxes.dtype != np.float32 or keypoints_xy.dtype != np.float32 \
            or keypoints_conf.dtype != np.float32:
        raise AssertionError("Wrong output dtypes")
    if boxes.shape != (2, 4) or keypoints_xy.shape != (2, COCO_KEYPOINT_COUNT, 2) \
            or keypoints_conf.shape != (2, COCO_KEYPOINT_COUNT):
        raise AssertionError("Wrong output shapes")

    height, width = frame_shape
    if boxes[:, [0, 2]].min() < 0 or boxes[:, [0, 2]].max() > width:
        raise AssertionError("Box x-coordinates not clipped to the frame width")
    if boxes[:, [1, 3]].min() < 0 or boxes[:, [1, 3]].max() > height:
        raise AssertionError("Box y-coordinates not clipped to the frame height")

    # The letterboxed image's own centre maps back near the original frame's centre.
    expected_cx, expected_cy = width / 2, height / 2
    actual_cx = (boxes[0, 0] + boxes[0, 2]) / 2
    actual_cy = (boxes[0, 1] + boxes[0, 3]) / 2
    if abs(actual_cx - expected_cx) > 2 or abs(actual_cy - expected_cy) > 2:
        raise AssertionError("Centred detection did not map back near the frame centre")


def check_postprocess_empty():
    """No detection above threshold returns correctly shaped empty arrays."""
    size = (416, 416)
    frame_shape = (480, 640)
    _, scale, pad = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), size)

    output = _make_output([_make_detection(100, 100, 40, 40, box_conf=0.05)])
    boxes, keypoints_xy, keypoints_conf = postprocess(
        output, scale, pad, frame_shape, conf_thres=0.25, iou_thres=0.45
    )

    if boxes.shape != (0, 4) or boxes.dtype != np.float32:
        raise AssertionError(f"Wrong empty boxes shape/dtype: {boxes.shape}, {boxes.dtype}")
    if keypoints_xy.shape != (0, COCO_KEYPOINT_COUNT, 2) or keypoints_xy.dtype != np.float32:
        raise AssertionError(f"Wrong empty keypoints_xy shape/dtype: {keypoints_xy.shape}")
    if keypoints_conf.shape != (0, COCO_KEYPOINT_COUNT) or keypoints_conf.dtype != np.float32:
        raise AssertionError(f"Wrong empty keypoints_conf shape/dtype: {keypoints_conf.shape}")


def check_face_regions_matches_ultralytics_path():
    """face_regions() called directly on TrtPose-style arrays (as blur_demo.py does for
    --backend trt) gives the same regions as going through an Ultralytics-style result
    (as regions_from_result does for --backend ultralytics), for the same numbers."""
    boxes = np.array([[50, 50, 150, 250]], dtype=np.float32)  # one tall person box
    keypoints_xy = np.zeros((1, COCO_KEYPOINT_COUNT, 2), dtype=np.float32)
    keypoints_conf = np.zeros((1, COCO_KEYPOINT_COUNT), dtype=np.float32)

    # Nose, left eye, right eye visible -> a "face" region.
    keypoints_xy[0, 0] = (90, 70)
    keypoints_xy[0, 1] = (85, 65)
    keypoints_xy[0, 2] = (95, 65)
    keypoints_conf[0, [0, 1, 2]] = 0.9

    direct_regions = face_regions(boxes, keypoints_xy, keypoints_conf)

    fake_result = FakeResult(boxes, keypoints_xy, keypoints_conf)
    via_result_regions = regions_from_result(fake_result)

    if direct_regions != via_result_regions:
        raise AssertionError(
            f"Regions differ between the trt and ultralytics code paths:\n"
            f"  direct:     {direct_regions}\n"
            f"  via result: {via_result_regions}"
        )
    if len(direct_regions) != 1 or direct_regions[0][1] != "face":
        raise AssertionError(f"Expected a single face region, got {direct_regions}")


def main():
    checks = [
        check_letterbox_shape_and_padding,
        check_letterbox_roundtrip,
        check_postprocess_confidence_filter,
        check_postprocess_nms,
        check_postprocess_scale_back_and_clip,
        check_postprocess_empty,
        check_face_regions_matches_ultralytics_path,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")

    print(f"\n{len(checks)} checks passed.")


if __name__ == "__main__":
    main()
