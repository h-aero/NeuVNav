"""
laps_line.py — HSV-based boundary line detection for LTA platforms.

Algorithm after: Hufen, F. (2019). Entwicklung einer autonomen Steuerung fuer
Unmanned Aerial Vehicles mithilfe von Methoden der Bildverarbeitung.
Master's thesis, Hochschule Anhalt.

Hufen demonstrated the first application of this CV pipeline to the h-aero(r)
LTA platform. This module reimplements the approach with LTA-specific tuning:
downward-facing camera, low-altitude flight, floor boundary detection.

ENFIELD WP2 -- funded by the European Union, grant No 101120657.
"""

import cv2
import numpy as np


# Default HSV range for a high-contrast floor boundary line.
# Tune HSV_LOWER / HSV_UPPER for your specific line color and lighting.
HSV_LOWER_DEFAULT = np.array([0,   0,   0])    # dark / black line
HSV_UPPER_DEFAULT = np.array([180, 255, 80])


class LineDetector:
    """
    Detects a boundary line in a camera frame using the HSV+Canny+Hough pipeline.

    Usage::

        detector = LineDetector()
        detected, cx, frame_w = detector.detect(frame)
        if detected:
            offset = cx - frame_w // 2  # positive = line right of center

    Parameters
    ----------
    hsv_lower : np.ndarray
        Lower HSV bound for the target line color.
    hsv_upper : np.ndarray
        Upper HSV bound for the target line color.
    blur_kernel : int
        GaussianBlur kernel size (odd number). Default 5.
    canny_lo : int
        Canny lower threshold. Default 50.
    canny_hi : int
        Canny upper threshold. Default 150.
    hough_threshold : int
        Minimum Hough votes to accept a line. Default 30.
    hough_min_length : int
        Minimum line segment length in pixels. Default 20.
    hough_max_gap : int
        Maximum gap between segments to join. Default 30.
    """

    def __init__(
        self,
        hsv_lower=None,
        hsv_upper=None,
        blur_kernel=5,
        canny_lo=50,
        canny_hi=150,
        hough_threshold=30,
        hough_min_length=20,
        hough_max_gap=30,
    ):
        self.hsv_lower = hsv_lower if hsv_lower is not None else HSV_LOWER_DEFAULT
        self.hsv_upper = hsv_upper if hsv_upper is not None else HSV_UPPER_DEFAULT
        self.blur_kernel = blur_kernel
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap

    def detect(self, frame):
        """
        Run the detection pipeline on a single BGR frame.

        Returns
        -------
        detected : bool
            True if at least one line segment was found.
        cx : float or None
            Horizontal center of detected line(s) in pixels. None if not detected.
        frame_width : int
            Width of the input frame (for offset calculation).
        """
        blurred = cv2.GaussianBlur(
            frame, (self.blur_kernel, self.blur_kernel), 0
        )
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        edges = cv2.Canny(mask, self.canny_lo, self.canny_hi)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        if lines is None:
            return False, None, frame.shape[1]

        # Average horizontal center across all detected segments
        xs = [(x1 + x2) / 2.0 for x1, y1, x2, y2 in lines[:, 0]]
        cx = float(np.mean(xs))
        return True, cx, frame.shape[1]

    def detect_with_debug(self, frame):
        """
        Like detect(), but also returns an annotated debug frame.
        Draws detected line segments in green.
        """
        detected, cx, fw = self.detect(frame)
        debug = frame.copy()

        blurred = cv2.GaussianBlur(
            frame, (self.blur_kernel, self.blur_kernel), 0
        )
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        edges = cv2.Canny(mask, self.canny_lo, self.canny_hi)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return detected, cx, fw, debug


def run_camera_loop(detector=None, camera_index=0):
    """
    Simple camera loop for live testing on hardware.
    Press 'q' to quit.

    Requires: picamera (RPi) or standard webcam.
    On RPi with picamera, replace VideoCapture with PiCamera capture loop.
    """
    if detector is None:
        detector = LineDetector()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected, cx, fw, debug = detector.detect_with_debug(frame)
        status = f"LINE cx={cx:.0f}" if detected else "NO LINE"
        cv2.putText(debug, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("NeuVNav laps_line", debug)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_loop()
