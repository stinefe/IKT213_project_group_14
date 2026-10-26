"""YOLO11-pose inference on the Jetson via TensorRT, as a drop-in for the
Ultralytics pose model used elsewhere in this project.

The letterbox/postprocess functions below are pure numpy + cv2 and have no
dependency on tensorrt or pycuda, so they can be imported and unit-tested on
a PC with neither package installed (see test_trt_pose.py). Only the TrtPose
class needs them, and it imports them lazily inside __init__ -- importing
this module on a machine without TensorRT still works.

Typical use on the Jetson:

    engine = TrtPose("models/yolo11n-pose.engine")
    boxes, keypoints_xy, keypoints_conf = engine(frame)
    regions = face_regions(boxes, keypoints_xy, keypoints_conf)
"""
import cv2
import numpy as np

COCO_KEYPOINT_COUNT = 17


def letterbox(frame, size=(416, 416), color=(114, 114, 114)):
    """Resize the frame to fit inside size=(width, height), preserving aspect ratio,
    and pad the rest with color.

    Returns (letterboxed_image, scale, (pad_left, pad_top)). scale and pad map pixel
    coordinates between the original frame and the letterboxed image; see
    scale_to_original / scale_to_letterboxed.
    """
    height, width = frame.shape[:2]
    target_w, target_h = size
    scale = min(target_w / width, target_h / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = target_w - new_w, target_h - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded, scale, (left, top)


def scale_to_original(points, scale, pad):
    """Map (..., 2) pixel coordinates from the letterboxed image back to the
    original frame."""
    left, top = pad
    out = np.array(points, dtype=np.float32, copy=True)
    out[..., 0] = (out[..., 0] - left) / scale
    out[..., 1] = (out[..., 1] - top) / scale
    return out


def scale_to_letterboxed(points, scale, pad):
    """Map (..., 2) pixel coordinates from the original frame to the letterboxed
    image (the inverse of scale_to_original)."""
    left, top = pad
    out = np.array(points, dtype=np.float32, copy=True)
    out[..., 0] = out[..., 0] * scale + left
    out[..., 1] = out[..., 1] * scale + top
    return out


def _xywh_to_xyxy(boxes_xywh):
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms(boxes, scores, iou_thres):
    """Greedy NMS over a single class. Returns kept indices, highest score first."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]

        xx1, yy1 = np.maximum(x1[i], x1[rest]), np.maximum(y1[i], y1[rest])
        xx2, yy2 = np.minimum(x2[i], x2[rest]), np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)

        order = rest[iou <= iou_thres]

    return np.array(keep, dtype=np.int64)


def postprocess(output, scale, pad, frame_shape, conf_thres=0.25, iou_thres=0.45,
                 num_keypoints=COCO_KEYPOINT_COUNT):
    """Decode one raw YOLO11-pose engine output into frame-pixel detections.

    output:      raw model output, shape (1, 5 + 3*num_keypoints, N) or (5+3*K, N):
                 box (cx, cy, w, h, conf) followed by (x, y, conf) per keypoint, all
                 in letterboxed-image pixels.
    scale, pad:  as returned by letterbox(), to map back onto the original frame.
    frame_shape: (height, width) of the original frame, for clipping.

    Returns (boxes, keypoints_xy, keypoints_conf): (N,4) xyxy, (N,K,2), (N,K) float32
    arrays, in the same layout as falldet.face_blur.regions_from_result expects.
    """
    output = np.asarray(output, dtype=np.float32)
    if output.ndim == 3:
        output = output[0]
    predictions = output.T  # (N, 5 + 3*num_keypoints)

    height, width = frame_shape
    empty = (np.empty((0, 4), dtype=np.float32),
              np.empty((0, num_keypoints, 2), dtype=np.float32),
              np.empty((0, num_keypoints), dtype=np.float32))

    predictions = predictions[predictions[:, 4] >= conf_thres]
    if len(predictions) == 0:
        return empty

    boxes_xyxy = _xywh_to_xyxy(predictions[:, :4])
    keep = _nms(boxes_xyxy, predictions[:, 4], iou_thres)
    if len(keep) == 0:
        return empty
    boxes_xyxy = boxes_xyxy[keep]

    keypoints = predictions[keep, 5:].reshape(-1, num_keypoints, 3)
    keypoints_xy, keypoints_conf = keypoints[:, :, :2], keypoints[:, :, 2]

    boxes_xyxy = scale_to_original(boxes_xyxy.reshape(-1, 2, 2), scale, pad).reshape(-1, 4)
    keypoints_xy = scale_to_original(keypoints_xy, scale, pad)

    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, width)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, height)
    keypoints_xy[:, :, 0] = np.clip(keypoints_xy[:, :, 0], 0, width)
    keypoints_xy[:, :, 1] = np.clip(keypoints_xy[:, :, 1], 0, height)

    return (boxes_xyxy.astype(np.float32),
            keypoints_xy.astype(np.float32),
            keypoints_conf.astype(np.float32))


class TrtPose(object):
    """YOLO11-pose TensorRT engine, callable like a single-image model.

    TensorRT and pycuda are imported here, inside __init__, not at module level,
    so letterbox()/postprocess() above stay importable without either package
    (e.g. for unit tests on a PC with no GPU).
    """

    def __init__(self, engine_path, input_size=416, conf_thres=0.25, iou_thres=0.45):
        import pycuda.autoinit  # noqa: F401 -- creates the CUDA context for this thread
        import pycuda.driver as cuda
        import tensorrt as trt

        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._cuda = cuda
        self._stream = cuda.Stream()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as engine_file, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(engine_file.read())
        self._context = self._engine.create_execution_context()

        self._bindings = []
        self._host = {}
        self._device = {}
        self._input_name = None
        self._output_name = None
        self._output_shape = None

        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            shape = tuple(self._engine.get_binding_shape(i))
            dtype = trt.nptype(self._engine.get_binding_dtype(i))

            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))
            self._host[name] = host_mem
            self._device[name] = device_mem

            if self._engine.binding_is_input(i):
                self._input_name = name
            else:
                self._output_name = name
                self._output_shape = shape

    def __call__(self, frame):
        """Run pose inference on one BGR frame. Returns the same (boxes, keypoints_xy,
        keypoints_conf) layout as regions_from_result expects."""
        letterboxed, scale, pad = letterbox(frame, (self.input_size, self.input_size))

        blob = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        np.copyto(self._host[self._input_name], blob.ravel())

        cuda = self._cuda
        cuda.memcpy_htod_async(self._device[self._input_name], self._host[self._input_name],
                                self._stream)
        self._context.execute_async_v2(self._bindings, self._stream.handle)
        cuda.memcpy_dtoh_async(self._host[self._output_name], self._device[self._output_name],
                                self._stream)
        self._stream.synchronize()

        output = self._host[self._output_name].reshape(self._output_shape)
        return postprocess(output, scale, pad, frame.shape[:2], self.conf_thres, self.iou_thres)
