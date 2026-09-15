"""OpenCV camera preprocessing and debug-image output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CameraProcessor:
    """Normalize MuJoCo RGB frames and persist perception diagnostics."""

    def __init__(self, config: dict[str, Any] | None = None, output_dir: str | Path = "outputs/perception") -> None:
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = int(self.config.get("width", 320))
        self.height = int(self.config.get("height", 240))

    def process_frame(self, frame: Any, camera_name: str = "overhead") -> np.ndarray:
        """Validate, resize, denoise, and return an RGB uint8 frame."""
        if frame is None:
            raise ValueError("frame cannot be None")
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("camera frame must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        self.save_debug_image(f"{camera_name}_rgb.png", image)
        return image

    def save_debug_image(self, filename: str, rgb_image: np.ndarray) -> Path:
        """Save an RGB image using OpenCV's BGR file-writing convention."""
        if not filename or rgb_image is None:
            raise ValueError("filename and image are required")
        path = self.output_dir / Path(filename).name
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"OpenCV could not write debug image: {path}")
        return path

    def save_detection_debug(self, camera_name: str, rgb_frame: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> dict[str, Path]:
        """Save RGB, detection-overlay, and segmentation images."""
        return {
            "rgb": self.save_debug_image(f"{camera_name}_rgb.png", rgb_frame),
            "detections": self.save_debug_image(f"{camera_name}_detected_objects.png", overlay),
            "segmentation": self.save_debug_image(f"{camera_name}_segmentation.png", mask),
        }

    def perceive(self, observation: dict[str, Any], model: Any, data: Any, camera_name: str = "overhead", detector: Any = None) -> Any:
        """Fuse a camera frame with simulator state into a :class:`SceneState`."""
        if not isinstance(observation, dict) or "camera_images" not in observation:
            raise ValueError("observation must contain camera_images")
        frame = self.process_frame(observation["camera_images"][camera_name], camera_name)
        if detector is None:
            from .object_detection import GroundTruthDetector
            detector = GroundTruthDetector(model, data, {**self.config, "width": self.width, "height": self.height})
        detections = detector.detect(frame, camera_name)
        from .object_detection import detections_to_scene_objects
        from .scene_state import SceneState
        from .segmentation import segmentation_from_detections

        overlay = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, detection.label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        segmentation = segmentation_from_detections(frame, detections)
        self.save_detection_debug(camera_name, frame, overlay, segmentation)
        table_body = model.body("table").id
        table_position = np.asarray(data.xpos[table_body], dtype=np.float32).tolist()
        return SceneState(
            detected_objects=detections_to_scene_objects(detections, camera_name),
            drawer_state=dict(observation.get("drawer_state", {})),
            table_state={"position": table_position, "body": "table"},
            robot_state={
                "joint_positions": observation.get("robot_joint_positions"),
                "joint_velocities": observation.get("robot_joint_velocities"),
                "task_state": observation.get("task_state", {}),
            },
            camera_name=camera_name,
            timestamp=float(observation.get("task_state", {}).get("step", 0)),
        )


def process_frame(frame: Any, config: dict[str, Any] | None = None) -> np.ndarray:
    """Backward-compatible functional camera preprocessing entry point."""
    return CameraProcessor(config).process_frame(frame)