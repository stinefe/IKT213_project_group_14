"""Checks that falldet.face_blur's blur actually destroys face detail (sharpness drops
by a large factor, measured with the Laplacian variance) and leaves the rest of the
frame untouched, for both a small region (dataset-video scale) and a large one
(close-up camera scale, the case the downscaled blur in face_blur.py targets).

Runs on a PC with only numpy and cv2.
"""
import numpy as np

from falldet.face_blur import FEATHER_FRACTION, blur_region

# Checkerboard squares much smaller than any blur kernel: high Laplacian variance
# before blurring, near zero after, regardless of region size.
CHECKER_SQUARE = 4


def checkerboard_frame(height, width):
    """A high-frequency uint8 BGR frame: black/white squares, so any real blur
    collapses its Laplacian variance."""
    yy, xx = np.indices((height, width))
    pattern = ((yy // CHECKER_SQUARE) + (xx // CHECKER_SQUARE)) % 2
    gray = (pattern * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def laplacian_variance(gray_or_bgr):
    """Sharpness measure: high for detailed/noisy content, near zero for flat/blurred."""
    import cv2
    gray = gray_or_bgr if gray_or_bgr.ndim == 2 else cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var()


def check_blur_destroys_detail(box, frame_shape, min_reduction_factor=50):
    """Laplacian variance inside box drops by at least min_reduction_factor."""
    height, width = frame_shape
    frame = checkerboard_frame(height, width)
    x1, y1, x2, y2 = (int(round(v)) for v in box)

    before = laplacian_variance(frame[y1:y2, x1:x2])
    if before < 1000:
        raise AssertionError(f"Checkerboard fixture is not detailed enough: var={before}")

    blurred = frame.copy()
    blur_region(blurred, box)
    after = laplacian_variance(blurred[y1:y2, x1:x2])

    reduction = before / max(after, 1e-6)
    if reduction < min_reduction_factor:
        raise AssertionError(
            f"Blur too weak for box {box} on a {width}x{height} frame: "
            f"variance {before:.0f} -> {after:.0f} (x{reduction:.1f}, "
            f"need >= x{min_reduction_factor})"
        )
    return blurred


def check_small_region_is_blurred():
    """A small, dataset-video-scale face region (320x240 frame)."""
    check_blur_destroys_detail(box=(120, 80, 160, 120), frame_shape=(240, 320))


def check_large_region_is_blurred():
    """A large, close-up region on a 1280x720 frame -- the case the downscaled blur
    in face_blur.py targets (previously a huge Gaussian kernel at full resolution)."""
    check_blur_destroys_detail(box=(400, 10, 700, 710), frame_shape=(720, 1280))


def check_outside_region_is_unchanged():
    """Pixels outside the box and its feather margin are not touched, for a shape
    that uses both branches: a small region (box shape) and a large one (ellipse)."""
    import cv2

    for shape, box, frame_shape in [
        ("box", (120, 80, 160, 120), (240, 320)),
        ("ellipse", (400, 10, 700, 710), (720, 1280)),
    ]:
        height, width = frame_shape
        frame = checkerboard_frame(height, width)
        x1, y1, x2, y2 = box

        blurred = frame.copy()
        blur_region(blurred, box, shape=shape)

        # Margin wide enough to clear the feather pad used for the ellipse shape (see
        # _blur_ellipse: pad = 3 * feather, feather = FEATHER_FRACTION * shorter side).
        box_w, box_h = x2 - x1, y2 - y1
        feather = max(2, round(FEATHER_FRACTION * min(box_w, box_h)))
        margin = 3 * feather + 2

        mask = np.ones((height, width), dtype=bool)
        mx1, my1 = max(x1 - margin, 0), max(y1 - margin, 0)
        mx2, my2 = min(x2 + margin, width), min(y2 + margin, height)
        mask[my1:my2, mx1:mx2] = False

        if not np.array_equal(blurred[mask], frame[mask]):
            raise AssertionError(
                f"Pixels outside the {shape} region (plus feather margin) changed"
            )


def main():
    checks = [
        check_small_region_is_blurred,
        check_large_region_is_blurred,
        check_outside_region_is_unchanged,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")

    print(f"\n{len(checks)} checks passed.")


if __name__ == "__main__":
    main()
