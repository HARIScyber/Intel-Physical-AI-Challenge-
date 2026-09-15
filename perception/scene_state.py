"""Typed multimodal scene state shared by perception and planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectedObject:
    """One object detection grounded in simulator or image evidence."""

    name: str
    object_class: str
    position: tuple[float, float, float]
    bounding_box: tuple[int, int, int, int]
    confidence: float
    camera: str
    segmentation_area: int = 0


@dataclass
class SceneState:
    """Complete perception output for one synchronized simulation observation."""

    detected_objects: list[DetectedObject] = field(default_factory=list)
    drawer_state: dict[str, Any] = field(default_factory=dict)
    table_state: dict[str, Any] = field(default_factory=dict)
    robot_state: dict[str, Any] = field(default_factory=dict)
    camera_name: str = "overhead"
    timestamp: float = 0.0

    @property
    def objects(self) -> dict[str, dict[str, Any]]:
        """Expose object records in the legacy dictionary shape."""
        return {item.name: asdict(item) for item in self.detected_objects}

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly scene-state mapping."""
        return {
            "detected_objects": [asdict(item) for item in self.detected_objects],
            "drawer_state": self.drawer_state,
            "table_state": self.table_state,
            "robot_state": self.robot_state,
            "camera_name": self.camera_name,
            "timestamp": self.timestamp,
        }