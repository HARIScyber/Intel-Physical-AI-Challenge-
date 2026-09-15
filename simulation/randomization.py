"""Reproducible domain randomization profiles for MuJoCo evaluation."""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

MODE_SCALE = {"easy": 0.5, "medium": 1.0, "hard": 1.75}
LANGUAGE_VARIANTS = {
    "easy": ("Set the dinner table.",),
    "medium": ("Set the dinner table.", "Please arrange the table for dinner."),
    "hard": ("Set the dinner table.", "Prepare the dining table and organize the place settings.", "Arrange everything needed for dinner."),
}


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy deterministic sources."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)


def mode_scale(mode: str) -> float:
    """Return the perturbation scale for an evaluation mode."""
    if mode not in MODE_SCALE:
        raise ValueError(f"evaluation mode must be one of {tuple(MODE_SCALE)}")
    return MODE_SCALE[mode]


def randomize_model(model: Any, rng: np.random.Generator, config: dict[str, Any], mode: str = "medium") -> dict[str, Any]:
    """Randomize physical/rendering parameters and return the applied record."""
    scale = mode_scale(mode)
    randomization = config.get("randomization", {})
    friction_range = _scaled_range(randomization.get("friction_range", [0.6, 1.2]), scale, 1.0)
    mass_range = _scaled_range(randomization.get("mass_range", [0.8, 1.2]), scale, 1.0)
    light_range = _scaled_range(randomization.get("light_intensity_range", [0.75, 1.25]), scale, 1.0)
    camera_range = float(randomization.get("camera_position_range", 0.04)) * scale
    size_range = _scaled_range(randomization.get("size_range", [0.9, 1.1]), scale, 1.0)
    color_range = float(randomization.get("color_jitter", 0.08)) * scale
    table_scale_range = _scaled_range(randomization.get("table_scale_range", [0.95, 1.05]), scale, 1.0)
    friction_values: list[float] = []
    mass_values: list[float] = []
    size_values: list[float] = []
    for geom_id in range(model.ngeom):
        friction = float(rng.uniform(*friction_range))
        model.geom_friction[geom_id, 0] = friction
        friction_values.append(friction)
        if _is_task_object_geom(model, geom_id):
            size = float(rng.uniform(*size_range))
            model.geom_size[geom_id] *= size
            size_values.append(size)
            model.geom_rgba[geom_id, :3] = np.clip(model.geom_rgba[geom_id, :3] + rng.uniform(-color_range, color_range, 3), 0.0, 1.0)
    for body_id in range(1, model.nbody):
        factor = float(rng.uniform(*mass_range))
        model.body_mass[body_id] *= factor
        model.body_inertia[body_id] *= factor
        mass_values.append(factor)
    for light_id in range(model.nlight):
        model.light_diffuse[light_id] *= float(rng.uniform(*light_range))
    table_scale = float(rng.uniform(*table_scale_range))
    for geom_id in range(model.ngeom):
        if model.geom(geom_id).name in {"tabletop", "table_front"}:
            model.geom_size[geom_id, :2] *= table_scale
    if camera_range > 0:
        model.cam_pos[:] += rng.uniform(-camera_range, camera_range, size=model.cam_pos.shape)
    background = np.clip(rng.uniform(0.08, 0.24, 3) * (1.0 + 0.15 * scale), 0.0, 1.0)
    model.geom_rgba[0, :3] = background
    return {"mode": mode, "friction": friction_values, "mass_scale": mass_values, "object_size_scale": size_values, "table_scale": table_scale, "camera_jitter": camera_range, "background_rgb": background.tolist()}


def randomize_initial_state(model: Any, data: Any, rng: np.random.Generator, mode: str = "medium") -> dict[str, Any]:
    """Randomize object poses, orientations, robot poses, and availability."""
    scale = mode_scale(mode)
    position_range = 0.08 * scale
    object_names = ["plate", "cup", "spoon", "fork", "napkin", "bowl"]
    positions: dict[str, list[float]] = {}
    orientations: dict[str, list[float]] = {}
    for name in object_names:
        joint_id = model.joint(f"{name}_free").id
        address = model.jnt_qposadr[joint_id]
        data.qpos[address:address + 3] += rng.uniform([-position_range, -position_range, 0.0], [position_range, position_range, 0.01 * scale])
        angle = float(rng.uniform(-np.pi * scale, np.pi * scale))
        data.qpos[address + 3:address + 7] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
        positions[name] = data.qpos[address:address + 3].astype(float).tolist()
        orientations[name] = data.qpos[address + 3:address + 7].astype(float).tolist()
    robot_positions: dict[str, list[float]] = {}
    for label in ("a", "b"):
        values: list[float] = []
        for index in range(1, 7):
            joint_id = model.joint(f"arm_{label}_joint_{index}").id
            address = model.jnt_qposadr[joint_id]
            data.qpos[address] = rng.uniform(-0.08 * scale, 0.08 * scale)
            values.append(float(data.qpos[address]))
        robot_positions[label] = values
    order = object_names.copy()
    rng.shuffle(order)
    available_count = len(order) if mode == "easy" else int(rng.integers(max(3, len(order) - int(scale)), len(order) + 1))
    available = order[:available_count]
    for name in object_names:
        if name not in available:
            body_id = model.body(name).id
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    model.geom_contype[geom_id] = 0
                    model.geom_conaffinity[geom_id] = 0
    return {"object_positions": positions, "object_orientations": orientations, "robot_initial_joints": robot_positions, "object_order": order, "available_objects": available, "unavailable_objects": [name for name in object_names if name not in available]}


def randomize_control_noise(rng: np.random.Generator, action: np.ndarray, config: dict[str, Any], mode: str = "medium") -> np.ndarray:
    """Apply bounded reproducible control noise to a command."""
    scale = mode_scale(mode)
    amount = float(config.get("randomization", {}).get("control_noise", 0.0)) * scale
    if amount <= 0:
        return np.asarray(action, dtype=np.float64).copy()
    return np.asarray(action, dtype=np.float64) + rng.normal(0.0, amount, size=np.asarray(action).shape)


def language_variant(rng: np.random.Generator, mode: str = "medium") -> str:
    """Choose a deterministic wording variant from the seeded generator."""
    variants = LANGUAGE_VARIANTS[mode]
    return variants[int(rng.integers(0, len(variants)))]


def _scaled_range(values: Any, scale: float, center: float) -> tuple[float, float]:
    """Scale a two-value range around its center."""
    low, high = (float(value) for value in values)
    return center + (low - center) * scale, center + (high - center) * scale


def _is_task_object_geom(model: Any, geom_id: int) -> bool:
    """Return whether a geometry belongs to a randomized tabletop object."""
    body_name = model.body(int(model.geom_bodyid[geom_id])).name
    return body_name in {"plate", "cup", "spoon", "fork", "napkin", "bowl"}