"""Camera configuration and frame capture boundary for MuJoCo."""

from dataclasses import dataclass
import logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    """Named camera settings passed to a renderer or OpenCV processor."""

    name: str = "overhead"
    width: int = 640
    height: int = 480
