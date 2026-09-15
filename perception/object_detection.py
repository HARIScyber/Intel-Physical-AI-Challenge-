"""Configurable object-detection interfaces and MuJoCo ground-truth backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .scene_state import DetectedObject


@dataclass(frozen=True)
class Detection:
    """One image-space detection."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mask: np.ndarray | None = None


class DetectorBackend(Protocol):
    """Replaceable detector contract for a learned vision model."""

    def detect(self, frame: np.ndarray, camera_name: str = "overhead") -> list[Detection]:
        """Detect objects in an RGB frame."""


class GroundTruthDetector:
    """Detect configured object bodies from MuJoCo segmentation render output."""

    def __init__(self, model: Any, data: Any, config: dict[str, Any] | None = None) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MuJoCo is required for ground-truth perception") from exc
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.config = config or {}
        self.object_names = tuple(self.config.get("object_names", ("plate", "cup", "spoon", "fork", "napkin", "bowl")))
        self.width = int(self.config.get("width", 320))
        self.height = int(self.config.get("height", 240))
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.renderer.enable_segmentation_rendering()
        self._geom_ids = self._object_geom_ids()

    def detect(self, frame: np.ndarray, camera_name: str = "overhead") -> list[Detection]:
        """Return detections whose boxes are computed from segmentation pixels."""
        del frame
        self.renderer.update_scene(self.data, camera=camera_name)
        segmentation = np.asarray(self.renderer.render())
        if segmentation.ndim != 3 or segmentation.shape[2] < 2:
            raise RuntimeError("MuJoCo segmentation renderer returned an unexpected image")
        detections: list[Detection] = []
        for name in self.object_names:
            geom_ids = self._geom_ids.get(name, set())
            mask = (segmentation[:, :, 0] == int(self.mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[:, :, 1], list(geom_ids))
            ys, xs = np.where(mask)
            body_id = self.model.body(name).id
            position = tuple(float(value) for value in self.data.xpos[body_id])
            if xs.size == 0:
                bbox = self._projected_bbox(name, camera_name)
                mask = None
            else:
                bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            detections.append(Detection(name, 1.0, bbox, position, mask))
        return detections

    def _object_geom_ids(self) -> dict[str, set[int]]:
        """Resolve object body names to their compiled MuJoCo geom IDs."""
        result: dict[str, set[int]] = {}
        for name in self.object_names:
            body_id = self.model.body(name).id
            result[name] = {geom_id for geom_id in range(self.model.ngeom) if self.model.geom_bodyid[geom_id] == body_id}
        return result

    def _projected_bbox(self, name: str, camera_name: str) -> tuple[int, int, int, int]:
        """Project a live body pose to pixels when its geometry is occluded."""
        camera = self.renderer.scene.camera[0]
        body_id = self.model.body(name).id
        point = np.asarray(self.data.xpos[body_id], dtype=np.float64)
        forward = np.asarray(camera.forward, dtype=np.float64)
        up = np.asarray(camera.up, dtype=np.float64)
        right = np.cross(forward, up)
        relative = point - np.asarray(camera.pos, dtype=np.float64)
        depth = float(np.dot(relative, forward))
        width = self.width
        height = self.height
        if depth <= 1e-6:
            return (0, 0, 1, 1)
        near = float(camera.frustum_near)
        configured_width = float(camera.frustum_width)
        if configured_width <= 1e-6:
            configured_width = 2.0 * float(camera.frustum_top) * width / height
        half_width = max(configured_width / 2.0, 1e-6)
        half_height = max(float(camera.frustum_top), 1e-6)
        pixel_x = width / 2.0 + np.dot(relative, right) / depth * near / half_width * width / 2.0
        pixel_y = height / 2.0 - np.dot(relative, up) / depth * near / half_height * height / 2.0
        geom_ids = self._geom_ids.get(name, set())
        extent = max((float(np.max(self.model.geom_size[geom_id])) for geom_id in geom_ids), default=0.03)
        radius_x = max(3.0, extent / depth * near / half_width * width / 2.0)
        radius_y = max(3.0, extent / depth * near / half_height * height / 2.0)
        x1 = int(np.clip(pixel_x - radius_x, 0, width - 1))
        y1 = int(np.clip(pixel_y - radius_y, 0, height - 1))
        x2 = int(np.clip(pixel_x + radius_x, x1 + 1, width))
        y2 = int(np.clip(pixel_y + radius_y, y1 + 1, height))
        return x1, y1, x2, y2


def detections_to_scene_objects(detections: list[Detection], camera_name: str) -> list[DetectedObject]:
    """Convert backend-neutral detections to typed scene objects."""
    return [DetectedObject(item.label, item.label, item.position, item.bbox, item.confidence, camera_name, int(np.count_nonzero(item.mask)) if item.mask is not None else 0) for item in detections]


def detect_objects(frame: Any, confidence_threshold: float = 0.5) -> list[Detection]:
    """Validate a frame for compatibility with future learned detector backends."""
    if frame is None or not 0 <= confidence_threshold <= 1:
        raise ValueError("frame must be provided and threshold must be in [0, 1]")
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    return []