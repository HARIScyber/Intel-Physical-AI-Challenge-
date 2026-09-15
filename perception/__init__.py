"""Perception interfaces for camera observations."""

from .camera_processor import CameraProcessor
from .object_detection import Detection, DetectorBackend, GroundTruthDetector
from .scene_state import DetectedObject, SceneState

__all__ = ["CameraProcessor", "Detection", "DetectedObject", "DetectorBackend", "GroundTruthDetector", "SceneState"]
