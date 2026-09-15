"""Task object definitions and MuJoCo state extraction helpers."""

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneObject:
    """A named object with a semantic category and 3D position."""

    name: str
    category: str
    position: tuple[float, float, float]


OBJECT_NAMES = ("plate", "cup", "spoon", "fork", "napkin", "bowl")


def collect_object_state(model: Any, data: Any) -> dict[str, dict[str, list[float]]]:
    """Read object poses from MuJoCo body state using the real model indices."""
    state: dict[str, dict[str, list[float]]] = {}
    for name in OBJECT_NAMES:
        try:
            body_id = model.body(name).id
        except (KeyError, ValueError):
            LOGGER.warning("Configured object body %s is missing", name)
            continue
        state[name] = {
            "position": np.asarray(data.xpos[body_id], dtype=np.float32).tolist(),
            "orientation": np.asarray(data.xquat[body_id], dtype=np.float32).tolist(),
        }
    return state
