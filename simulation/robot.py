"""Simulation-side SO-101 arm metadata and joint state helpers."""

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class SimulatedArm:
    """Configuration-independent description of one simulated SO-101 arm."""

    name: str
    joint_positions: list[float] = field(default_factory=lambda: [0.0] * 6)

    def reset(self) -> None:
        """Reset all joints to a neutral pose."""
        self.joint_positions = [0.0] * len(self.joint_positions)


def arm_joint_names(label: str) -> tuple[str, ...]:
    """Return the six actuated joint names for one simulated SO-101 arm."""
    if label not in {"a", "b"}:
        raise ValueError("arm label must be 'a' or 'b'")
    return tuple(f"arm_{label}_joint_{index}" for index in range(1, 7))


def collect_arm_state(model: Any, data: Any, label: str) -> dict[str, list[float]]:
    """Read joint positions and velocities through MuJoCo joint addresses."""
    positions: list[float] = []
    velocities: list[float] = []
    for name in arm_joint_names(label):
        joint_id = model.joint(name).id
        qpos_address = model.jnt_qposadr[joint_id]
        dof_address = model.jnt_dofadr[joint_id]
        positions.append(float(data.qpos[qpos_address]))
        velocities.append(float(data.qvel[dof_address]))
    return {"joint_positions": positions, "joint_velocities": velocities}
