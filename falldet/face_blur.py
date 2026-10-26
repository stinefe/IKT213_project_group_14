"""Face blur for live pose previews (camera or video), driven by YOLO pose keypoints.

The blur is applied to a display copy of the frame. Pose inference must run on the
original frame, so blurring never changes the extracted keypoints.

Typical use with an Ultralytics pose result:

    result = model.predict(frame, verbose=False)[0]
    shown = blur_faces_from_result(frame, result)
"""
import math

import cv2
import numpy as np

# COCO keypoint indices used by YOLO11 pose
FACE_IDS = (0, 1, 2, 3, 4)   # nose, left eye, right eye, left ear, right ear
SHOULDER_IDS = (5, 6)
HIP_IDS = (11, 12)

FACE_KPT_CONF = 0.3       # below the 0.5 analysis threshold: over-blurring is the safe error
BODY_KPT_CONF = 0.5       # shoulders/hips, used to find the head of a lying person
FALLBACK_FRACTION = 0.2   # part of the person's box blurred when face keypoints are missing
MIN_FACE_FRACTION = 0.1   # minimum face-box half-size, relative to the box's longer side
BLUR_SHAPE = "ellipse"    # "ellipse": soft oval over the face; "box": hard rectangle
FEATHER_FRACTION = 0.25   # width of the soft edge, relative to the region's shorter side

# A region is blurred at a shrunk size, then resized back up: a small blur on a small
# copy anonymises at least as strongly as a huge kernel at full resolution (no detail
# survives the downscale), for a fraction of the cost. Regions already at or below this
# size are blurred at their native size, unchanged from before this was added.
_BLUR_WORK_MAX_SIDE = 40

# cv2's own masked blend (in C++) is far cheaper at full resolution than the equivalent
# numpy broadcast, which has to expand the single-channel mask over 3 channels itself.
# Guarded in case a given OpenCV build lacks it; falls back to the numpy version below.
_HAS_BLEND_LINEAR = hasattr(cv2, "blendLinear")


def _visible(xy, conf, ids, threshold):
    """Pixel coordinates of the listed keypoints whose confidence meets the threshold."""
    if xy is None or conf is None:
        return np.empty((0, 2), dtype=np.float32)
    ids = list(ids)
    return xy[ids][conf[ids] >= threshold]


def face_box_from_keypoints(person_box, xy, conf):
    """Square around the visible nose/eye/ear keypoints, or None if none are visible."""
    points = _visible(xy, conf, FACE_IDS, FACE_KPT_CONF)
    if len(points) == 0:
        return None

    x1, y1, x2, y2 = person_box
    longer_side = max(x2 - x1, y2 - y1)
    cx, cy = points.mean(axis=0)
    # Ear-to-ear or eye-to-ear distance approximates head size. The minimum keeps the
    # box large enough when only one or two points are visible, or the person is small.
    spread = float(np.ptp(points, axis=0).max())
    half = max(spread, MIN_FACE_FRACTION * longer_side)
    return (cx - half, cy - half, cx + half, cy + half)


def fallback_boxes(person_box, xy=None, conf=None):
    """Areas to blur when no face keypoint is visible.

    Upright person: the top FALLBACK_FRACTION of the box.
    Wider than tall (e.g. lying after a fall): the head is at the left or right end,
    chosen from shoulders vs hips; if those are not visible either, blur both ends.
    """
    x1, y1, x2, y2 = person_box
    width, height = x2 - x1, y2 - y1

    if width <= height:
        return [(x1, y1, x2, y1 + FALLBACK_FRACTION * height)]

    band = FALLBACK_FRACTION * width
    left = (x1, y1, x1 + band, y2)
    right = (x2 - band, y1, x2, y2)

    shoulders = _visible(xy, conf, SHOULDER_IDS, BODY_KPT_CONF)
    hips = _visible(xy, conf, HIP_IDS, BODY_KPT_CONF)
    if len(shoulders) and len(hips):
        return [left] if shoulders[:, 0].mean() < hips[:, 0].mean() else [right]
    return [left, right]


def face_regions(boxes, keypoints_xy=None, keypoints_conf=None):
    """Return (box, kind) pairs to blur for every detected person.

    boxes:          (N, 4) person boxes, x1 y1 x2 y2 in pixels
    keypoints_xy:   (N, 17, 2) keypoints in pixels (not normalized), or None
    keypoints_conf: (N, 17) keypoint confidences, or None
    kind is "face" (built from face keypoints) or "fallback".
    """
    regions = []
    for i, box in enumerate(boxes):
        xy = keypoints_xy[i] if keypoints_xy is not None else None
        conf = keypoints_conf[i] if keypoints_conf is not None else None

        face = face_box_from_keypoints(box, xy, conf)
        if face is not None:
            regions.append((face, "face"))
        else:
            regions.extend((area, "fallback") for area in fallback_boxes(box, xy, conf))
    return regions


def _blur_kernel(region_width, region_height):
    # Kernel grows with the region (capped by the downscale below) so a small region
    # is blurred as strongly as the small faces in the 320x240 dataset videos. Must
    # be odd.
    return max(31, (max(region_width, region_height) // 2) | 1)


def _downscale_for_blur(region):
    """Shrink region so its longer side is at most _BLUR_WORK_MAX_SIDE, ready for a
    cheap small-kernel blur. Returns (small_region, scale); scale is 1.0 (region
    returned as is, no resize) when it is already that small or smaller.
    """
    height, width = region.shape[:2]
    scale = min(1.0, _BLUR_WORK_MAX_SIDE / max(width, height))
    if scale >= 1.0:
        return region, 1.0
    small_w = max(1, int(round(width * scale)))
    small_h = max(1, int(round(height * scale)))
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return small, scale


def _blur_box(frame, box):
    """Hard-edged rectangular blur, in place.

    Blurs a downscaled copy of the region and resizes the result back up, instead of
    running a huge kernel at full resolution (see _downscale_for_blur).
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(math.floor(x1), 0), max(math.floor(y1), 0)
    x2, y2 = min(math.ceil(x2), width), min(math.ceil(y2), height)
    if x2 <= x1 or y2 <= y1:
        return

    region = frame[y1:y2, x1:x2]
    small, scale = _downscale_for_blur(region)
    kernel = _blur_kernel(small.shape[1], small.shape[0])
    blurred = cv2.GaussianBlur(small, (kernel, kernel), 0)
    if scale < 1.0:
        blurred = cv2.resize(blurred, (region.shape[1], region.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = blurred


def _blur_ellipse(frame, box):
    """Oval blur that fills the box, with a soft edge fading outward, in place.

    Everything inside the oval is fully blurred; the fade only extends outside it,
    so the soft edge never uncovers part of the face. The blur and the feather mask
    are both built on a downscaled copy of the work area and resized back up together
    (see _downscale_for_blur), so there is no large-kernel blur or full-resolution
    mask-blurring, however large the region.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return

    feather = max(2, int(round(FEATHER_FRACTION * min(box_w, box_h))))
    pad = 3 * feather  # room for the fade around the oval

    # Work area: the box plus padding, clipped to the frame
    ax1, ay1 = max(math.floor(x1) - pad, 0), max(math.floor(y1) - pad, 0)
    ax2, ay2 = min(math.ceil(x2) + pad, width), min(math.ceil(y2) + pad, height)
    if ax2 <= ax1 or ay2 <= ay1:
        return

    area = frame[ay1:ay2, ax1:ax2]
    area_h, area_w = area.shape[:2]
    small, scale = _downscale_for_blur(area)
    small_h, small_w = small.shape[:2]

    kernel = _blur_kernel(small_w, small_h)
    small_blurred = cv2.GaussianBlur(small, (kernel, kernel), 0)

    # Ellipse geometry at the same reduced scale as small, so mask and blur line up
    # after both are resized back to area_w x area_h below.
    centre = (int(round(((x1 + x2) / 2 - ax1) * scale)),
              int(round(((y1 + y2) / 2 - ay1) * scale)))
    axes = (max(1, int(round(box_w / 2 * scale))), max(1, int(round(box_h / 2 * scale))))
    small_feather = max(1.0, feather * scale)
    grown = (axes[0] + int(round(small_feather)), axes[1] + int(round(small_feather)))

    # Soft mask: a slightly larger oval, smoothed, so it is ~1 at the face oval
    # and fades to 0 outside it. The hard oval keeps the inside fully covered.
    hard = np.zeros((small_h, small_w), dtype=np.float32)
    cv2.ellipse(hard, centre, axes, 0, 0, 360, 1.0, -1)
    soft = np.zeros_like(hard)
    cv2.ellipse(soft, centre, grown, 0, 0, 360, 1.0, -1)
    soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=small_feather / 2)
    small_mask = np.maximum(hard, soft)

    if scale < 1.0:
        blurred = cv2.resize(small_blurred, (area_w, area_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(small_mask, (area_w, area_h), interpolation=cv2.INTER_LINEAR)
    else:
        blurred, mask = small_blurred, small_mask

    # blurred*mask + area*(1-mask): the only full-resolution work left, now that the
    # blur itself and its mask are both computed at the downscaled size above.
    if _HAS_BLEND_LINEAR and frame.dtype == np.uint8:
        area[:] = cv2.blendLinear(blurred, area, mask, 1.0 - mask)
    else:
        mask = mask[..., None]
        diff = blurred.astype(np.float32) - area.astype(np.float32)
        mixed = area.astype(np.float32) + diff * mask
        area[:] = np.clip(np.rint(mixed), 0, 255).astype(frame.dtype)


def blur_region(frame, box, shape=None):
    """Blur one region of the frame in place; the region is clipped to the frame.

    shape: "ellipse" (soft oval) or "box" (hard rectangle); defaults to BLUR_SHAPE.
    """
    if (shape or BLUR_SHAPE) == "box":
        _blur_box(frame, box)
    else:
        _blur_ellipse(frame, box)


def blur_regions(frame, regions, shape=None):
    """Return a blurred copy of the frame; the input frame is not modified."""
    out = frame.copy()
    for box, _ in regions:
        blur_region(out, box, shape)
    return out


def regions_from_result(result):
    """Face regions for all people in one Ultralytics pose result."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints
    if keypoints is None or keypoints.conf is None:
        return face_regions(boxes)
    return face_regions(
        boxes,
        keypoints.xy.cpu().numpy(),
        keypoints.conf.cpu().numpy(),
    )


def blur_faces_from_result(frame, result):
    """Blurred copy of the frame, using the people and keypoints in the result."""
    return blur_regions(frame, regions_from_result(result))