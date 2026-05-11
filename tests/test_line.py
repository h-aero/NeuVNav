"""Tests for laps_line.py — synthetic images, no camera needed."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from python.laps_line import LineDetector


def make_black_line_image(width=640, height=480, y=240, thickness=12):
    """White image with a horizontal black line."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.line(img, (50, y), (width - 50, y), (0, 0, 0), thickness)
    return img


def test_detects_black_line():
    detector = LineDetector()
    img = make_black_line_image()
    detected, cx, fw = detector.detect(img)
    assert detected is True
    assert cx is not None


def test_line_center_approx_correct():
    detector = LineDetector()
    img = make_black_line_image(width=640, y=240)
    detected, cx, fw = detector.detect(img)
    assert detected
    assert fw == 640
    assert abs(cx - 320) < 80  # line roughly centered


def test_no_line_on_blank_image():
    detector = LineDetector()
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
    detected, cx, fw = detector.detect(blank)
    assert detected is False
    assert cx is None


def test_frame_width_returned():
    detector = LineDetector()
    img = make_black_line_image(width=320, height=240)
    _, _, fw = detector.detect(img)
    assert fw == 320


def test_debug_mode_returns_frame():
    detector = LineDetector()
    img = make_black_line_image()
    detected, cx, fw, debug = detector.detect_with_debug(img)
    assert debug.shape == img.shape


if __name__ == "__main__":
    test_detects_black_line()
    test_line_center_approx_correct()
    test_no_line_on_blank_image()
    test_frame_width_returned()
    test_debug_mode_returns_frame()
    print("All line detection tests passed.")
