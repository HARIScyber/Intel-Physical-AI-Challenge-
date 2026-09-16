"""Authoritative task verification system for the dinner-table manipulation task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from language.command_schema import TaskCommand
from policy.base_policy import OBJECT_HALF_HEIGHTS, TABLE_TOP


@dataclass(frozen=True)
class ObjectVerification:
    """Verification result for a single required object."""

    required: bool
    located: bool = False
    grasped: bool = False
    transported: bool = False
    released: bool = False
    placed: bool = False
    verified: bool = False
    reason: str = ""
    grasp_verified: bool = False
    placement_verified: bool = False
    manipulation_history: dict[str, bool] = field(default_factory=dict)


@dataclass
class TaskVerificationResult:
    """Complete structured task verification result."""

    success: bool = False
    drawer_open: bool = False
    objects: dict[str, ObjectVerification] = field(default_factory=dict)
    collision_free: bool = True
    bimanual_actions_verified: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary."""
        return {
            "success": self.success,
            "drawer_open": self.drawer_open,
            "objects": {name: asdict(obj) for name, obj in self.objects.items()},
            "collision_free": self.collision_free,
            "bimanual_actions_verified": self.bimanual_actions_verified,
            "failure_reason": self.failure_reason,
        }


class TaskVerifier:
    """Authoritative task verification for the dinner-table task.

    This is the SINGLE source of truth for task completion.
    All components (env, planner, policy, demo script) must use this verifier.
    """

    REQUIRED_OBJECTS = ("plate", "cup", "spoon", "fork")

    PLACEMENT_REGIONS = {
        "plate": {"center": np.array([0.0, 0.0, TABLE_TOP + OBJECT_HALF_HEIGHTS["plate"]]), "pos_tol": 0.08, "ori_tol": 0.15},
        "cup": {"center": np.array([0.2, 0.15, TABLE_TOP + OBJECT_HALF_HEIGHTS["cup"]]), "pos_tol": 0.08, "ori_tol": 0.15},
        "spoon": {"center": np.array([-0.15, 0.2, TABLE_TOP + OBJECT_HALF_HEIGHTS["spoon"]]), "pos_tol": 0.06, "ori_tol": 0.2},
        "fork": {"center": np.array([-0.15, -0.2, TABLE_TOP + OBJECT_HALF_HEIGHTS["fork"]]), "pos_tol": 0.06, "ori_tol": 0.2},
    }

    REQUIRED_HISTORY_STATES = [
        "located", "approached", "pregrasp", "grasped", "lifted",
        "transported", "preplaced", "released", "placed", "verified",
    ]

    def __init__(self, environment: Any, task_command: Any = None) -> None:
        self.env = environment
        self.task_command = task_command
        self._manipulation_history: dict[str, set[str]] = {}

    def _get_required_objects(self) -> tuple[str, ...]:
        """Get required objects from task command or defaults."""
        if self.task_command and self.task_command.objects:
            objects = tuple(obj for obj in self.task_command.objects if obj not in {"table", "drawer"})
            if objects:
                return objects
        return self.REQUIRED_OBJECTS

    def verify(self, manipulation_history: dict[str, set[str]] | None = None) -> TaskVerificationResult:
        """Perform complete task verification.

        Args:
            manipulation_history: Dict mapping object_name -> set of completed states

        Returns:
            TaskVerificationResult with all verification details
        """
        if manipulation_history is not None:
            self._manipulation_history = manipulation_history

        result = TaskVerificationResult()
        result.drawer_open = self.env._drawer_open()
        if not result.drawer_open:
            result.failure_reason = "Drawer is not open"
            return result

        required_objects = self._get_required_objects()

        for obj_name in required_objects:
            result.objects[obj_name] = ObjectVerification(required=True)

        for obj_name in required_objects:
            try:
                self.env.model.body(obj_name).id
            except (KeyError, ValueError):
                result.objects[obj_name].reason = f"Required object '{obj_name}' does not exist in simulation"
                result.failure_reason = result.objects[obj_name].reason
                return result
            result.objects[obj_name].located = True

        result.collision_free = not self.env.check_collision()
        if not result.collision_free:
            result.failure_reason = "Unsafe collision detected"
            return result

        all_verified = True
        for obj_name in required_objects:
            obj_ver = result.objects[obj_name]
            history = self._manipulation_history.get(obj_name, set())

            obj_ver.manipulation_history = {state: state in history for state in self.REQUIRED_HISTORY_STATES}

            for state in self.REQUIRED_HISTORY_STATES:
                if state not in history:
                    obj_ver.reason = f"Missing manipulation state: {state}"
                    all_verified = False
                    break

            if not all_verified:
                result.failure_reason = f"Object '{obj_name}': {obj_ver.reason}"
                return result

            obj_ver.grasped = "grasped" in history
            obj_ver.transported = "transported" in history
            obj_ver.released = "released" in history
            obj_ver.placed = "placed" in history
            obj_ver.verified = "verified" in history

            body_id = self.env.model.body(obj_name).id
            position = np.asarray(self.env.data.xpos[body_id], dtype=np.float64)
            quat = np.asarray(self.env.data.xquat[body_id], dtype=np.float64)

            region = self.PLACEMENT_REGIONS.get(obj_name)
            if region:
                pos_error = float(np.linalg.norm(position - region["center"]))
                if pos_error <= region["pos_tol"]:
                    obj_ver.placement_verified = True
                else:
                    obj_ver.reason = f"Position error {pos_error:.4f} exceeds tolerance {region['pos_tol']}"
                    all_verified = False
                    break

                if obj_name in ("spoon", "fork"):
                    local_y = np.array([0.0, 1.0, 0.0])
                    world_y = self._rotate_vector(local_y, quat)
                    z_axis = np.array([0.0, 0.0, 1.0])
                    ori_alignment = float(np.dot(world_y, z_axis))
                    if ori_alignment < 0.7:
                        obj_ver.reason = f"Orientation misaligned: y-up alignment = {ori_alignment:.3f}"
                        all_verified = False
                        break

            obj_ver.verified = (
                obj_ver.placement_verified
                and obj_ver.grasped
                and obj_ver.transported
                and obj_ver.released
                and obj_ver.placed
            )

            if not obj_ver.verified:
                all_verified = False
                result.failure_reason = f"Object '{obj_name}' verification failed"
                break

        if self.task_command and self.task_command.requires_bimanual:
            result.bimanual_actions_verified = any(
                "bimanual" in state or "handoff" in state or "cooperative" in state
                for obj_history in self._manipulation_history.values()
                for state in obj_history
            )
            if not result.bimanual_actions_verified:
                result.failure_reason = "Bimanual coordination required but not verified"
                all_verified = False

        result.success = all_verified
        if result.success:
            result.failure_reason = ""

        return result

    @staticmethod
    def _rotate_vector(vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion (w, x, y, z)."""
        w, x, y, z = quat
        q_vec = np.array([x, y, z])
        return vector + 2.0 * np.cross(q_vec, np.cross(q_vec, vector) + w * vector)

    def record_manipulation_state(self, object_name: str, state: str) -> None:
        """Record a manipulation state for an object."""
        if object_name not in self._manipulation_history:
            self._manipulation_history[object_name] = set()
        self._manipulation_history[object_name].add(state)

    def get_manipulation_history(self) -> dict[str, set[str]]:
        """Get current manipulation history."""
        return self._manipulation_history.copy()


def verify_task(environment: Any, task_command: Any = None, manipulation_history: dict[str, set[str]] | None = None) -> TaskVerificationResult:
    """Convenience function for task verification."""
    verifier = TaskVerifier(environment, task_command)
    return verifier.verify(manipulation_history)