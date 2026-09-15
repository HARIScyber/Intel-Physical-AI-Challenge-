"""Collision-safe coordination for two simulated SO-101 arms."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .arm_a import ArmAController
from .arm_b import ArmBController
from .arm_controller import Pose

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - setup_check covers this
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)


class BimanualController:
    """Coordinate atomic dual-arm moves against the live MuJoCo model."""

    def __init__(self, environment: Any, config: dict[str, Any] | None = None) -> None:
        if not hasattr(environment, "model") or not hasattr(environment, "data"):
            raise ValueError("environment must expose a live MuJoCo model and data")
        self.environment = environment
        self.model = environment.model
        self.data = environment.data
        self.config = config or {}
        self.arm_a = ArmAController(environment, config=self.config)
        self.arm_b = ArmBController(environment, config=self.config)

    def move_both(
        self,
        targets_a: Sequence[float],
        targets_b: Sequence[float],
        duration: float = 1.0,
        execute: bool = True,
    ) -> dict[str, Any]:
        """Validate and command both arms as one collision-checked operation."""
        joints_a = self.arm_a._validate_joint_targets(targets_a, duration)
        joints_b = self.arm_b._validate_joint_targets(targets_b, duration)
        self._assert_pair_safe(joints_a, joints_b)
        self.arm_a._write_joint_targets(joints_a)
        self.arm_b._write_joint_targets(joints_b)
        if execute:
            physics_steps = max(1, int(round(self.arm_a.control_dt / self.model.opt.timestep)))
            mujoco.mj_step(self.model, self.data, nstep=physics_steps)
        return self.get_state()

    def handoff(
        self,
        object_pose: Pose | dict[str, Sequence[float]],
        from_arm: str = "arm_a",
        to_arm: str = "arm_b",
        duration: float = 1.0,
    ) -> dict[str, Any]:
        """Perform a guarded source-to-target handoff preparation.

        The source arm approaches and closes first. The target arm then solves
        the same pose and is only allowed to approach if the pairwise contact
        preflight remains safe. This deliberately refuses an unsafe overlap
        instead of silently allowing inter-arm collision.
        """
        source, target = self._arms(from_arm, to_arm)
        source.move_to_pose(object_pose, duration=duration, execute=False)
        source.close_gripper(execute=True)
        target_joints = target.inverse_kinematics(object_pose)
        self._assert_pair_safe(source.joint_positions(), target_joints)
        target.move_joint(target_joints, duration=duration, execute=False)
        target.close_gripper(execute=True)
        source.open_gripper(execute=True)
        return self.get_state()

    def synchronized_reach(
        self,
        pose_a: Pose | dict[str, Sequence[float]],
        pose_b: Pose | dict[str, Sequence[float]],
        duration: float = 1.0,
        execute: bool = True,
    ) -> dict[str, Any]:
        """Solve both IK problems and execute them through one safety gate."""
        return self.move_both(
            self.arm_a.inverse_kinematics(pose_a),
            self.arm_b.inverse_kinematics(pose_b),
            duration=duration,
            execute=execute,
        )

    def parallel_actions(self, actions: dict[str, dict[str, Any]], duration: float = 1.0) -> dict[str, Any]:
        """Execute two joint or pose actions atomically after pairwise preflight."""
        if set(actions) != {"arm_a", "arm_b"}:
            raise ValueError("parallel_actions requires arm_a and arm_b actions")
        targets: list[np.ndarray] = []
        for arm_name in ("arm_a", "arm_b"):
            arm = self.arm_a if arm_name == "arm_a" else self.arm_b
            command = actions[arm_name]
            if "pose" in command:
                targets.append(arm.inverse_kinematics(command["pose"]))
            elif "joint_targets" in command:
                targets.append(np.asarray(command["joint_targets"], dtype=np.float64))
            else:
                raise ValueError(f"{arm_name} action requires pose or joint_targets")
        return self.move_both(targets[0], targets[1], duration=duration)

    def safe_sequence(self, sequence: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute a sequence and stop immediately when a safety check fails."""
        if not sequence:
            raise ValueError("safe_sequence requires at least one action")
        results: list[dict[str, Any]] = []
        for action in sequence:
            if action.get("type") == "move_both":
                result = self.move_both(action["arm_a"], action["arm_b"], duration=float(action.get("duration", 1.0)))
            elif action.get("type") == "open_grippers":
                self.arm_a.open_gripper()
                self.arm_b.open_gripper()
                result = self.get_state()
            elif action.get("type") == "close_grippers":
                self.arm_a.close_gripper()
                self.arm_b.close_gripper()
                result = self.get_state()
            else:
                raise ValueError(f"unsupported safe action type: {action.get('type')}")
            results.append(result)
        return results

    def stop(self) -> dict[str, Any]:
        """Stop and hold both arms."""
        self.arm_a.stop()
        self.arm_b.stop()
        return self.get_state()

    def execute_action(self, action: dict[str, Any], duration: float = 0.05) -> dict[str, Any]:
        """Safely execute one policy action through both simulated arms."""
        if not isinstance(action, dict):
            raise ValueError("policy action must be a mapping")
        targets_a = np.asarray(action.get("arm_a", self.arm_a.joint_positions()), dtype=np.float64)
        targets_b = np.asarray(action.get("arm_b", self.arm_b.joint_positions()), dtype=np.float64)
        joints_a = self.arm_a._validate_joint_targets(targets_a, duration)
        joints_b = self.arm_b._validate_joint_targets(targets_b, duration)
        self._assert_pair_safe(joints_a, joints_b)
        self.arm_a._write_joint_targets(joints_a)
        self.arm_b._write_joint_targets(joints_b)
        self.data.ctrl[self.arm_a._gripper_actuator_ids[0]] = 0.02 - 0.04 * float(np.clip(action.get("gripper_a", 0.0), 0.0, 1.0))
        self.data.ctrl[self.arm_a._gripper_actuator_ids[1]] = -0.02 + 0.04 * float(np.clip(action.get("gripper_a", 0.0), 0.0, 1.0))
        self.data.ctrl[self.arm_b._gripper_actuator_ids[0]] = 0.02 - 0.04 * float(np.clip(action.get("gripper_b", 0.0), 0.0, 1.0))
        self.data.ctrl[self.arm_b._gripper_actuator_ids[1]] = -0.02 + 0.04 * float(np.clip(action.get("gripper_b", 0.0), 0.0, 1.0))
        if "drawer" in action and action["drawer"] is not None:
            self.data.ctrl[self.model.actuator("drawer_motor").id] = float(np.clip(action["drawer"], -0.35, 0.02))
        physics_steps = max(1, int(round(duration / self.model.opt.timestep)))
        mujoco.mj_step(self.model, self.data, nstep=physics_steps)
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Return separate state for both arms."""
        return {"arm_a": self.arm_a.get_state(), "arm_b": self.arm_b.get_state()}

    def _assert_pair_safe(self, targets_a: np.ndarray, targets_b: np.ndarray) -> None:
        """Evaluate both target poses in MuJoCo before changing live state."""
        snapshot = (self.data.qpos.copy(), self.data.qvel.copy())
        try:
            for arm, targets in ((self.arm_a, targets_a), (self.arm_b, targets_b)):
                for name, value in zip(arm.joint_names, targets):
                    self.data.qpos[arm._qpos_address(name)] = value
            mujoco.mj_forward(self.model, self.data)
            if self.environment.check_collision():
                raise RuntimeError("bimanual target configuration would cause a collision")
        finally:
            self.data.qpos[:], self.data.qvel[:] = snapshot
            mujoco.mj_forward(self.model, self.data)

    def _arms(self, from_arm: str, to_arm: str) -> tuple[ArmAController | ArmBController, ArmAController | ArmBController]:
        """Resolve distinct source and target arm names."""
        if from_arm == to_arm or {from_arm, to_arm} != {"arm_a", "arm_b"}:
            raise ValueError("from_arm and to_arm must be distinct arm_a/arm_b names")
        return (self.arm_a if from_arm == "arm_a" else self.arm_b, self.arm_a if to_arm == "arm_a" else self.arm_b)