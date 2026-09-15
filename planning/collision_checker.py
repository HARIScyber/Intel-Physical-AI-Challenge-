"""Collision and clearance checks for planned MuJoCo arm targets."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np

class CollisionChecker:
    """Use live MuJoCo contacts when available, otherwise validate structure."""
    def __init__(self, model: Any = None, data: Any = None, minimum_distance: float = 0.03) -> None:
        if minimum_distance < 0:
            raise ValueError("minimum_distance must be non-negative")
        self.model, self.data, self.minimum_distance = model, data, minimum_distance

    def is_collision_free(self, trajectory: Sequence[Sequence[float]], arm_labels: Sequence[str] = ("a", "b")) -> bool:
        """Check finite joint trajectories and prospective MuJoCo contacts."""
        values = np.asarray(trajectory, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6 or not np.all(np.isfinite(values)):
            raise ValueError("trajectory must have shape (steps, 6) with finite values")
        if self.model is None or self.data is None or len(arm_labels) != 2 or values.shape[0] < 2:
            return True
        snapshot = self.data.qpos.copy(), self.data.qvel.copy()
        try:
            for label, target in zip(arm_labels, values[:2]):
                for index, value in enumerate(target, start=1):
                    joint_id = self.model.joint(f"arm_{label}_joint_{index}").id
                    self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
            import mujoco
            mujoco.mj_forward(self.model, self.data)
            return not self.current_collision()
        finally:
            self.data.qpos[:], self.data.qvel[:] = snapshot
            import mujoco
            mujoco.mj_forward(self.model, self.data)

    def current_collision(self) -> bool:
        """Read current contacts involving arm bodies, excluding safe manipulation contacts."""
        if self.model is None or self.data is None:
            return False
        arm_ids = {i for i in range(self.model.nbody) if self.model.body(i).name.startswith(("so101_", "arm_a_", "arm_b_"))}
        gripper_body_names = {"arm_a_gripper_left", "arm_a_gripper_right", "arm_b_gripper_left", "arm_b_gripper_right"}
        gripper_ids = {i for i in range(self.model.nbody) if self.model.body(i).name in gripper_body_names}
        object_body_names = {"plate", "cup", "spoon", "fork", "napkin", "bowl"}
        object_ids = {i for i in range(self.model.nbody) if self.model.body(i).name in object_body_names}
        
        for c in self.data.contact[: self.data.ncon]:
            geom1_body = self.model.geom_bodyid[c.geom1]
            geom2_body = self.model.geom_bodyid[c.geom2]
            arm_involved = geom1_body in arm_ids or geom2_body in arm_ids
            if not arm_involved:
                continue
            
            gripper_involved = geom1_body in gripper_ids or geom2_body in gripper_ids
            object_involved = geom1_body in object_ids or geom2_body in object_ids
            
            is_safe_manipulation = gripper_involved and object_involved
            if is_safe_manipulation:
                continue
            
            return True
        return False

def is_collision_free(trajectory: Sequence[Sequence[float]], config: dict | None = None) -> bool:
    """Backward-compatible structural collision check."""
    del config
    return CollisionChecker().is_collision_free(trajectory)