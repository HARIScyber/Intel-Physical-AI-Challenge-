"""Hardware-independent MuJoCo controller for one simulated SO-101 arm."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - setup_check covers this
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None

from simulation.robot import arm_joint_names

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pose:
    """Cartesian end-effector pose in world coordinates."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


class ArmController:
    """Control one six-joint simulated arm through MuJoCo position actuators."""

    def __init__(self, name: str, model: Any = None, data: Any = None, config: dict[str, Any] | None = None) -> None:
        if name not in {"arm_a", "arm_b"}:
            raise ValueError("name must be 'arm_a' or 'arm_b'")
        if model is not None and hasattr(model, "model"):
            environment = model
            model, data = environment.model, environment.data
            self.environment = environment
        else:
            self.environment = None
        if model is None or data is None:
            raise ValueError("a live MuJoCo model and data are required")
        self.name = name
        self.label = name[-1]
        self.model = model
        self.data = data
        self.config = config or {}
        self.joint_names = arm_joint_names(self.label)
        self.gripper_joint_names = (f"arm_{self.label}_gripper_left_slide", f"arm_{self.label}_gripper_right_slide")
        self.site_name = f"arm_{self.label}_grasp_site"
        self.joint_limits = self._read_joint_limits()
        self.velocity_limits = np.full(6, float(self.config.get("max_joint_velocity", 4.0)), dtype=np.float64)
        self.control_dt = float(self.config.get("control_dt", 0.05))
        self._actuator_ids = tuple(self._actuator_id(f"arm_{self.label}_motor_{index}") for index in range(1, 7))
        self._gripper_actuator_ids = tuple(
            self._actuator_id(f"arm_{self.label}_gripper_{side}_motor") for side in ("left", "right")
        )

    def move_joint(self, joint_targets: Sequence[float], duration: float = 1.0, execute: bool = True) -> dict[str, Any]:
        """Move toward six joint targets after limit, velocity, and collision checks."""
        targets = self._validate_joint_targets(joint_targets, duration)
        self._assert_safe_target(targets)
        self._write_joint_targets(targets)
        if execute:
            self._step()
        return self.get_state()

    def move_to_pose(self, pose: Pose | dict[str, Sequence[float]], duration: float = 1.0, execute: bool = True) -> dict[str, Any]:
        """Solve numerical IK for a Cartesian pose and command the resulting joints."""
        return self.move_joint(self.inverse_kinematics(pose), duration=duration, execute=execute)

    def open_gripper(self, execute: bool = True) -> dict[str, Any]:
        """Command both physical gripper fingers to their open limits."""
        self._write_gripper(0.0)
        if execute:
            self._step(settle_seconds=0.25)
        return self.get_state()

    def close_gripper(self, execute: bool = True) -> dict[str, Any]:
        """Command both physical gripper fingers to their closed limits."""
        self._write_gripper(1.0)
        if execute:
            self._step(settle_seconds=0.25)
        return self.get_state()

    def stop(self) -> dict[str, Any]:
        """Hold current joint positions and clear current joint velocities."""
        self._write_joint_targets(self.joint_positions())
        self.data.qvel[[self._dof_address(name) for name in self.joint_names]] = 0.0
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Return separate joint, velocity, pose, and gripper state."""
        return {
            "name": self.name,
            "joint_positions": self.joint_positions().tolist(),
            "joint_velocities": self.joint_velocities().tolist(),
            "end_effector_pose": self.forward_kinematics().__dict__,
            "gripper": self.gripper_state(),
        }

    def forward_kinematics(self, joint_positions: Sequence[float] | None = None) -> Pose:
        """Return the grasp-site pose for current or temporary joint positions."""
        snapshot = self._set_temporary_joints(joint_positions) if joint_positions is not None else None
        try:
            site_id = self.model.site(self.site_name).id
            orientation = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(orientation, self.data.site_xmat[site_id])
            return Pose(tuple(self.data.site_xpos[site_id]), tuple(orientation))
        finally:
            self._restore_temporary(snapshot)

    def inverse_kinematics(self, pose: Pose | dict[str, Sequence[float]], max_iterations: int = 80, initial_guess: np.ndarray | None = None) -> np.ndarray:
        """Solve position-and-orientation IK with MuJoCo site Jacobians."""
        target = self._coerce_pose(pose)
        current = initial_guess.copy() if initial_guess is not None else self.joint_positions().copy()
        target_position = np.asarray(target.position, dtype=np.float64)
        target_quaternion = np.asarray(target.orientation, dtype=np.float64)
        site_id = self.model.site(self.site_name).id
        for _ in range(max_iterations):
            snapshot = self._set_temporary_joints(current)
            try:
                position_error = target_position - self.data.site_xpos[site_id]
                current_quaternion = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(current_quaternion, self.data.site_xmat[site_id])
                orientation_error = np.zeros(3, dtype=np.float64)
                mujoco.mju_subQuat(orientation_error, target_quaternion, current_quaternion)
                error = np.concatenate((position_error, orientation_error))
                if np.linalg.norm(error) < 1e-4:
                    break
                jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.data, jacobian_position, jacobian_rotation, site_id)
                columns = [self._dof_address(name) for name in self.joint_names]
                jacobian = np.vstack((jacobian_position[:, columns], jacobian_rotation[:, columns]))
                step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-3 * np.eye(6), error)
                current = np.clip(current + 0.1 * step, self.joint_limits[:, 0], self.joint_limits[:, 1])
            finally:
                self._restore_temporary(snapshot)
        if np.linalg.norm(self._pose_error(self.forward_kinematics(current), target)) > 0.03:
            raise ValueError(f"IK did not converge for {self.name}")
        return current

    def joint_positions(self) -> np.ndarray:
        """Read six joint positions from MuJoCo qpos."""
        return np.asarray([self.data.qpos[self._qpos_address(name)] for name in self.joint_names], dtype=np.float64)

    def joint_velocities(self) -> np.ndarray:
        """Read six joint velocities from MuJoCo qvel."""
        return np.asarray([self.data.qvel[self._dof_address(name)] for name in self.joint_names], dtype=np.float64)

    def gripper_state(self) -> dict[str, float]:
        """Return finger positions and normalized opening, where one is open."""
        left = float(self.data.qpos[self._qpos_address(self.gripper_joint_names[0])])
        right = float(self.data.qpos[self._qpos_address(self.gripper_joint_names[1])])
        opening = np.clip((left - right) / 0.04, 0.0, 1.0)
        return {"left_position": left, "right_position": right, "opening": float(opening)}

    def _validate_joint_targets(self, targets: Sequence[float], duration: float) -> np.ndarray:
        """Validate shape, finite values, joint limits, and requested velocity."""
        if duration <= 0:
            raise ValueError("duration must be positive")
        values = np.asarray(targets, dtype=np.float64).reshape(-1)
        if values.size != 6 or not np.all(np.isfinite(values)):
            raise ValueError("joint targets must contain six finite values")
        if np.any(values < self.joint_limits[:, 0]) or np.any(values > self.joint_limits[:, 1]):
            raise ValueError("joint target exceeds a configured joint limit")
        if np.any(np.abs(values - self.joint_positions()) / duration > self.velocity_limits):
            raise ValueError("joint target exceeds a configured velocity limit")
        return values

    def _assert_safe_target(self, targets: np.ndarray) -> None:
        """Reject targets that create arm-scene or inter-arm contacts."""
        snapshot = self._set_temporary_joints(targets)
        try:
            if self._has_arm_collision():
                raise RuntimeError(f"unsafe collision configuration for {self.name}")
        finally:
            self._restore_temporary(snapshot)

    def _write_joint_targets(self, targets: np.ndarray) -> None:
        """Write validated joint targets to real MuJoCo position actuators."""
        for actuator_id, target in zip(self._actuator_ids, targets):
            self.data.ctrl[actuator_id] = target

    def _write_gripper(self, opening: float) -> None:
        """Write opposing finger position targets to physical slide actuators."""
        if not 0.0 <= opening <= 1.0:
            raise ValueError("gripper opening must be in [0, 1]")
        self.data.ctrl[self._gripper_actuator_ids[0]] = 0.02 - 0.04 * opening
        self.data.ctrl[self._gripper_actuator_ids[1]] = -0.02 + 0.04 * opening

    def _step(self, settle_seconds: float | None = None) -> None:
        """Advance live MuJoCo state by a control interval or settling window."""
        seconds = settle_seconds if settle_seconds is not None else self.control_dt
        physics_steps = max(1, int(round(seconds / self.model.opt.timestep)))
        mujoco.mj_step(self.model, self.data, nstep=physics_steps)

    def _has_arm_collision(self) -> bool:
        """Inspect contacts involving any articulated arm body, excluding safe manipulation contacts."""
        arm_body_ids = {
            body_id for body_id in range(self.model.nbody)
            if self.model.body(body_id).name.startswith(("so101_", "arm_a_", "arm_b_"))
        }
        gripper_body_names = {
            "arm_a_gripper_left", "arm_a_gripper_right",
            "arm_b_gripper_left", "arm_b_gripper_right"
        }
        gripper_body_ids = {
            body_id for body_id in range(self.model.nbody)
            if self.model.body(body_id).name in gripper_body_names
        }
        object_body_names = {
            "plate", "cup", "spoon", "fork", "napkin", "bowl"
        }
        object_body_ids = {
            body_id for body_id in range(self.model.nbody)
            if self.model.body(body_id).name in object_body_names
        }
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body_a = self.model.geom_bodyid[contact.geom1]
            body_b = self.model.geom_bodyid[contact.geom2]
            
            arm_involved = body_a in arm_body_ids or body_b in arm_body_ids
            if not arm_involved:
                continue
            
            gripper_a = body_a in gripper_body_ids
            gripper_b = body_b in gripper_body_ids
            object_a = body_a in object_body_ids
            object_b = body_b in object_body_ids
            
            is_safe_manipulation = (gripper_a and object_b) or (gripper_b and object_a)
            if is_safe_manipulation:
                continue
            
            return True
        return False

    def _read_joint_limits(self) -> np.ndarray:
        """Read joint limits directly from the compiled MuJoCo model."""
        return np.asarray([self.model.jnt_range[self.model.joint(name).id] for name in self.joint_names], dtype=np.float64)

    def _actuator_id(self, name: str) -> int:
        """Resolve an actuator name and fail clearly when model-incompatible."""
        try:
            return self.model.actuator(name).id
        except (KeyError, ValueError) as exc:
            raise ValueError(f"MuJoCo model is missing actuator {name}") from exc

    def _qpos_address(self, joint_name: str) -> int:
        """Resolve a joint qpos address."""
        return int(self.model.jnt_qposadr[self.model.joint(joint_name).id])

    def _dof_address(self, joint_name: str) -> int:
        """Resolve a joint velocity address."""
        return int(self.model.jnt_dofadr[self.model.joint(joint_name).id])

    def _set_temporary_joints(self, joints: Sequence[float] | None) -> tuple[np.ndarray, np.ndarray] | None:
        """Temporarily set arm qpos and return qpos/qvel snapshots."""
        if joints is None:
            return None
        snapshot = (self.data.qpos.copy(), self.data.qvel.copy())
        for name, value in zip(self.joint_names, joints):
            self.data.qpos[self._qpos_address(name)] = value
        mujoco.mj_forward(self.model, self.data)
        return snapshot

    def _restore_temporary(self, snapshot: tuple[np.ndarray, np.ndarray] | None) -> None:
        """Restore a temporary MuJoCo state snapshot."""
        if snapshot is not None:
            self.data.qpos[:], self.data.qvel[:] = snapshot
            mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _coerce_pose(pose: Pose | dict[str, Sequence[float]]) -> Pose:
        """Normalize a pose dataclass or mapping."""
        if isinstance(pose, Pose):
            return pose
        try:
            return Pose(tuple(float(value) for value in pose["position"]), tuple(float(value) for value in pose["orientation"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("pose requires position[3] and orientation[4]") from exc

    @staticmethod
    def _pose_error(current: Pose, target: Pose) -> np.ndarray:
        """Return a compact position-and-quaternion-vector error."""
        return np.concatenate((np.subtract(target.position, current.position), np.subtract(target.orientation[1:], current.orientation[1:])))