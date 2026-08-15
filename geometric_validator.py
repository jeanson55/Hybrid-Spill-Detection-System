"""
Geometric Consistency Validator
Enforces kinematic plausibility of detections:
  - IoU persistence across frames
  - Centroid displacement constraints
  - Aspect ratio and area thresholds
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    bbox: tuple          # (x1, y1, x2, y2) in pixels
    confidence: float
    frame_id: int
    centroid: tuple = field(init=False)
    area: float = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.area = max(0.0, (x2 - x1) * (y2 - y1))


def compute_iou(boxA: tuple, boxB: tuple) -> float:
    """Intersection over Union between two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


class GeometricValidator:
    """
    Maintains a short history of detections and validates new detections
    against geometric plausibility constraints derived from the thesis
    (Section 3.3, Table 3.1).
    """

    # --- Constraint parameters (Table 3.1) ---
    MIN_AREA_PX: float = 500          # minimum bounding-box area (pixels²)
    MAX_AREA_PX: float = 500_000      # maximum bounding-box area
    MIN_ASPECT_RATIO: float = 0.1     # w/h or h/w
    MAX_CENTROID_DISP: float = 120.0  # max centroid shift per frame (pixels)
    MIN_IOU_PERSIST: float = 0.15     # minimum IoU with prior frame box
    HISTORY_LEN: int = 5              # frames kept in history

    def __init__(self):
        self._history: deque = deque(maxlen=self.HISTORY_LEN)

    def validate(self, det: Detection) -> tuple[bool, str]:
        """
        Returns (passed: bool, reason: str).
        reason is non-empty only on rejection.
        """
        x1, y1, x2, y2 = det.bbox
        w = x2 - x1
        h = y2 - y1

        # 1. Area check
        if det.area < self.MIN_AREA_PX:
            return False, f"area_too_small ({det.area:.0f} px²)"
        if det.area > self.MAX_AREA_PX:
            return False, f"area_too_large ({det.area:.0f} px²)"

        # 2. Aspect ratio check
        if h > 0:
            ar = w / h
            if ar < self.MIN_ASPECT_RATIO or ar > (1 / self.MIN_ASPECT_RATIO):
                return False, f"aspect_ratio_invalid ({ar:.2f})"

        # 3. History-based checks
        if self._history:
            prev = self._history[-1]

            # Centroid displacement
            dx = det.centroid[0] - prev.centroid[0]
            dy = det.centroid[1] - prev.centroid[1]
            displacement = np.sqrt(dx**2 + dy**2)
            if displacement > self.MAX_CENTROID_DISP:
                return False, f"centroid_jump ({displacement:.1f} px)"

            # IoU persistence — only checked if same-ish region persists
            iou = compute_iou(det.bbox, prev.bbox)
            # We allow new detections (no prior match) but flag abrupt teleports
            if displacement > 30 and iou < self.MIN_IOU_PERSIST:
                return False, f"iou_persistence_fail (IoU={iou:.3f})"

        self._history.append(det)
        return True, ""

    def reset(self):
        self._history.clear()
