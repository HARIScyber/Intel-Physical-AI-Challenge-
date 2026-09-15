"""Segmentation backends and OpenCV visualizations."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any

from .object_detection import Detection


def segmentation_from_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Build a color segmentation visualization from detector masks or boxes."""
    if frame is None or frame.ndim != 3:
        raise ValueError("frame must be an RGB image")
    visualization = np.zeros_like(frame, dtype=np.uint8)
    for index, detection in enumerate(detections, start=1):
        color = ((37 * index) % 255, (97 * index) % 255, (173 * index) % 255)
        if detection.mask is not None:
            visualization[detection.mask] = color
        else:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(visualization, (x1, y1), (x2, y2), color, thickness=-1)
    return cv2.addWeighted(frame, 0.35, visualization, 0.65, 0.0)


def segment_scene(frame: Any, config: dict[str, Any] | None = None) -> np.ndarray:
    """Return an empty mask for the replaceable learned segmentation backend."""
    del config
    if frame is None or not hasattr(frame, "shape"):
        raise ValueError("frame must expose a shape")
    height, width = frame.shape[:2]
    return np.zeros((height, width), dtype=np.uint8)