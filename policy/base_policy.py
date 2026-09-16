"""Policy contracts and deterministic scripted baseline for MuJoCo."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from planning.action_sequence import Action, ActionSequence

LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Metrics collected during one scripted baseline episode."""

    task_success: bool = False
    step_count: int = 0
    episode_duration_seconds: float = 0.0
    collisions: int = 0
    failed_grasps: int = 0
    successful_placements: int = 0
    actions_attempted: int = 0
    handoffs: int = 0
    left_arm_actions: int = 0
    right_arm_actions: int = 0
    bimanual_actions: int = 0
    single_arm_actions: int = 0
    recovery_attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metrics."""
        return self.__dict__.copy()


class BasePolicy(ABC):
    """Stable policy interface independent of a VLA model."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def predict(self, observation: Any, instruction: str | ActionSequence | None = None) -> Any:
        """Map camera/state observation and language/task input to robot action."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset policy-internal queues and episode state."""

    @abstractmethod
    def load_checkpoint(self, checkpoint: str | None = None) -> None:
        """Load optional model weights from a local path or Hub identifier."""
        raise NotImplementedError


OBJECT_HALF_HEIGHTS = {
    "plate": 0.018,
    "cup": 0.075,
    "spoon": 0.012,
    "fork": 0.012,
    "napkin": 0.006,
    "bowl": 0.045,
}

OBJECT_GRASP_ORIENTATIONS = {
    "spoon": (0.0, 0.0, 1.0, 0.0),
    "fork": (0.0, 0.0, 1.0, 0.0),
    "plate": (0.0, 0.0, 1.0, 0.0),
    "cup": (1.0, 0.0, 0.0, 0.0),
    "napkin": (0.0, 0.0, 1.0, 0.0),
    "bowl": (0.0, 0.0, 1.0, 0.0),
}

ARM_ASSIGNMENT = {
    "spoon": "a",
    "fork": "a",
    "plate": "a",
    "cup": "b",
    "napkin": "a",
    "bowl": "b",
    "drawer": "a",
}

OBJECT_ARM_MAP = ARM_ASSIGNMENT.copy()

TABLE_CENTER = np.array([0.0, 0.0, 0.75], dtype=np.float64)
TABLE_HEIGHT = 0.75
TABLE_TOP = 0.75
DRAWER_HANDLE = np.array([-0.92, -0.87, 0.95], dtype=np.float64)
APPROACH_OFFSET = np.array([0.0, 0.0, 0.08], dtype=np.float64)
TRANSPORT_HEIGHT = np.array([0.0, 0.0, 0.05], dtype=np.float64)
SAFE_POSE = np.array([0.0, -0.25, 0.45, 0.0, -0.2, 0.0], dtype=np.float64)

# Arm base positions in the world frame
ARM_A_BASE = np.array([-0.92, 0.05, 0.8], dtype=np.float64)
ARM_B_BASE = np.array([0.92, 0.05, 0.8], dtype=np.float64)
ARM_LENGTH = 0.98

# IK configuration
IK_POSITION_TOLERANCE = 0.03
IK_ORIENTATION_TOLERANCE = 0.05
IK_MAX_ITERATIONS = 100
IK_MAX_RANDOM_SAMPLES = 1000
IK_RANDOM_SAMPLE_THRESHOLD = 0.05
IK_FALLBACK_THRESHOLD = 0.20


class IKResult:
    """Structured result from inverse kinematics computation."""

    success: bool
    joints: np.ndarray
    position_error: float
    orientation_error: float
    method: str
    attempts: int
    failure_reason: str

    def __init__(
        self,
        success: bool,
        joints: np.ndarray,
        position_error: float,
        orientation_error: float,
        method: str,
        attempts: int,
        failure_reason: str = "",
    ) -> None:
        self.success = success
        self.joints = joints
        self.position_error = position_error
        self.orientation_error = orientation_error
        self.method = method
        self.attempts = attempts
        self.failure_reason = failure_reason

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""
        return {
            "success": self.success,
            "position_error": self.position_error,
            "orientation_error": self.orientation_error,
            "method": self.method,
            "attempts": self.attempts,
            "failure_reason": self.failure_reason,
        }


class ScriptedPolicy(BasePolicy):
    """Deterministic phase-aware baseline that emits MuJoCo action commands."""

    def __init__(self, config: dict[str, Any] | None = None, model: Any = None, data: Any = None) -> None:
        super().__init__(config)
        self.metrics = BaselineMetrics()
        self._action_index = 0
        self._sequence: ActionSequence | None = None
        self.model = model
        self.data = data
        self._held_object: str | None = None
        self._arm_controllers: dict[str, Any] = {}
        self._object_positions: dict[str, np.ndarray] = {}
        if model is not None and data is not None:
            self._init_controllers()

    def _init_controllers(self) -> None:
        try:
            from control.arm_a import ArmAController
            from control.arm_b import ArmBController
            self._arm_controllers = {
                "a": ArmAController(self.model, self.data, self.config),
                "b": ArmBController(self.model, self.data, self.config),
            }
            LOGGER.info("Arm controllers initialized: %s", list(self._arm_controllers.keys()))
        except Exception as exc:
            LOGGER.warning("Arm controllers unavailable for scripted policy: %s", exc)

    def reset(self, action_sequence: ActionSequence | None = None) -> None:
        """Start a deterministic episode with a fixed action sequence."""
        if action_sequence is not None:
            if not action_sequence.actions:
                raise ValueError("scripted policy requires a non-empty action sequence")
            self._sequence = action_sequence
        self._action_index = 0
        self.metrics = BaselineMetrics()
        self._held_object = None
        self._object_positions = {}

    def predict(self, observation: Any, instruction: str | ActionSequence | None = None) -> dict[str, Any]:
        """Emit one bounded arm/drawer command for the next semantic action."""
        seq = None
        scene_state = observation
        if isinstance(observation, ActionSequence):
            seq = observation
            scene_state = type("Scene", (), {"detected_objects": []})()
        if isinstance(instruction, ActionSequence):
            seq = instruction
        if seq is not None and seq is not self._sequence:
            self.reset(seq)
        if self._sequence is None:
            raise ValueError("call reset or provide action_sequence before predict")
        if scene_state is None:
            raise ValueError("scene_state is required")
        self._update_object_positions(scene_state)
        if self._action_index >= len(self._sequence.actions):
            return self._command("hold")
        action = self._sequence.actions[self._action_index]
        self._action_index += 1
        self.metrics.actions_attempted += 1
        return self._command(action.name, action)

    def load_checkpoint(self, checkpoint: str | None = None) -> None:
        """Scripted policies have no checkpoint to load."""
        if checkpoint:
            LOGGER.info("Ignoring checkpoint for scripted policy: %s", checkpoint)

    def complete(self, scene_state: Any, step_count: int, duration_seconds: float, collision_count: int) -> dict[str, Any]:
        """Finalize metrics from the simulator's authoritative state."""
        task_state = getattr(scene_state, "robot_state", {}).get("task_state", {})
        self.metrics.task_success = bool(task_state.get("complete", False))
        self.metrics.step_count = int(step_count)
        self.metrics.episode_duration_seconds = float(duration_seconds)
        self.metrics.collisions = int(collision_count)
        return self.metrics.as_dict()

    def _update_object_positions(self, scene_state: Any) -> None:
        detected = getattr(scene_state, "detected_objects", [])
        self._object_positions = {}
        for obj in detected:
            name = getattr(obj, "name", "")
            pos = getattr(obj, "position", None)
            if name and pos is not None:
                self._object_positions[name] = np.asarray(pos, dtype=np.float64)

    def _object_position(self, name: str) -> np.ndarray:
        """Get the object's position on the table surface for grasp targeting.

        Uses the object's current x, y from MuJoCo but sets z to the table
        surface height plus the object's half-height. This prevents fallen
        objects (with free joints) from producing unreachable grasp targets.
        """
        if self.data is not None:
            try:
                body_id = self.model.body(name).id
                current_pos = np.asarray(self.data.xpos[body_id], dtype=np.float64)
                current_quat = np.asarray(self.data.xquat[body_id], dtype=np.float64)
                LOGGER.debug("_object_position %s body_id=%d pos=%s quat=%s", name, body_id, current_pos.tolist(), current_quat.tolist())
                table_pos = current_pos.copy()
                half_height = OBJECT_HALF_HEIGHTS.get(name, 0.02)
                table_pos[2] = TABLE_TOP + half_height
                if name in {"spoon", "fork"}:
                    result = self._handle_grasp_target(body_id, table_pos, local_offsets=((0.0, -0.09, 0.0), (0.0, 0.09, 0.0)))
                    arm_label = self._arm_for_object(name)
                    site_name = f"arm_{arm_label}_grasp_site"
                    try:
                        site_id = self.model.site(site_name).id
                        ee_target = self.data.site_xpos[site_id].tolist()
                        ee_orient = np.empty(4, dtype=np.float64)
                        mujoco.mju_mat2Quat(ee_orient, self.data.site_xmat[site_id])
                        grasp_quat = OBJECT_GRASP_ORIENTATIONS.get(name, (0,0,0,1))
                        LOGGER.info("GRASP_TARGET obj=%s world_pos=%s world_quat=%s local_offset=(0.0,±0.09,0.0) arm=%s ee_target=%s ee_orient=%s grasp_quat=%s",
                                    name, current_pos.tolist(), current_quat.tolist(), arm_label, ee_target, ee_orient.tolist(), grasp_quat)
                    except Exception:
                        pass
                    return result
                if name in {"plate", "cup"}:
                    result = table_pos.copy()
                    arm_label = self._arm_for_object(name)
                    grasp_quat = OBJECT_GRASP_ORIENTATIONS.get(name, (0,0,0,1))
                    LOGGER.info("GRASP_TARGET obj=%s world_pos=%s world_quat=%s arm=%s grasp_quat=%s",
                                name, current_pos.tolist(), current_quat.tolist(), arm_label, grasp_quat)
                    return result
                return table_pos
            except Exception as exc:
                LOGGER.debug("_object_position %s exception: %s", name, exc)
        pos = self._object_positions.get(name)
        if pos is None:
            half_height = OBJECT_HALF_HEIGHTS.get(name, 0.02)
            return np.array([0.0, 0.0, TABLE_TOP + half_height], dtype=np.float64)
        return pos

    def _handle_grasp_target(self, body_id: int, base_pos: np.ndarray, local_offsets: tuple[tuple[float, float, float], tuple[float, float, float]]) -> np.ndarray:
        """Compute the world-frame grasp target from object pose and local offsets.

        Determines the closest arm base, rotates local offsets by the object's
        quaternion, and returns the best candidate in world coordinates.
        """
        if mujoco is None:
            return base_pos
        best_pos = base_pos.copy()
        best_dist = float("inf")
        closest_arm_base = ARM_A_BASE if np.linalg.norm(base_pos - ARM_A_BASE) <= np.linalg.norm(base_pos - ARM_B_BASE) else ARM_B_BASE
        quat = np.asarray(self.data.xquat[body_id], dtype=np.float64)
        for local in local_offsets:
            rotated = np.zeros(3, dtype=np.float64)
            mujoco.mju_rotVecQuat(rotated, np.array(local, dtype=np.float64), quat)
            candidate = base_pos + rotated
            dist = float(np.linalg.norm(candidate - closest_arm_base))
            if dist < best_dist:
                best_dist = dist
                best_pos = candidate
        LOGGER.debug("_handle_grasp_target body_id=%d base_pos=%s closest_arm=%s best_pos=%s best_dist=%.4f",
                      body_id, base_pos.tolist(), "a" if np.allclose(closest_arm_base, ARM_A_BASE) else "b", best_pos.tolist(), best_dist)
        return best_pos

    def _arm_for_object(self, name: str) -> str:
        """Determine which arm should grasp an object based on reachability.

        Uses the table-surface position (not the fallen position) to determine
        arm assignment. Falls back to OBJECT_ARM_MAP if reachability is unclear.
        """
        default = OBJECT_ARM_MAP.get(name, "a")
        pos = self._object_position(name)
        dist_a = float(np.linalg.norm(pos - ARM_A_BASE))
        dist_b = float(np.linalg.norm(pos - ARM_B_BASE))
        a_reachable = dist_a <= ARM_LENGTH * 0.95
        b_reachable = dist_b <= ARM_LENGTH * 0.95
        if a_reachable and not b_reachable:
            LOGGER.debug("_arm_for_object %s -> arm_a (a_reachable=%s, b_reachable=%s)", name, a_reachable, b_reachable)
            return "a"
        if b_reachable and not a_reachable:
            LOGGER.debug("_arm_for_object %s -> arm_b (a_reachable=%s, b_reachable=%s)", name, a_reachable, b_reachable)
            return "b"
        if a_reachable and b_reachable:
            result = "a" if dist_a <= dist_b else "b"
            LOGGER.debug("_arm_for_object %s -> arm_%s (dist_a=%.4f, dist_b=%.4f)", name, result, dist_a, dist_b)
            return result
        LOGGER.debug("_arm_for_object %s -> arm_%s (default, both unreachable)", name, default)
        return default

    def _ik_target(self, arm_label: str, world_position: np.ndarray, offset: np.ndarray | None = None, object_name: str | None = None) -> IKResult:
        """Solve IK for a Cartesian pose with structured validation.

        Returns an IKResult with success status, joint positions, and error metrics.
        Does not silently accept bad IK solutions.
        """
        target_pos = world_position + (offset if offset is not None else np.zeros(3, dtype=np.float64))
        controller = self._arm_controllers.get(arm_label)
        if controller is None:
            LOGGER.warning("_ik_target arm=%s has no controller", arm_label)
            return IKResult(False, np.zeros(6), float("inf"), float("inf"), "NO_CONTROLLER", 0, "No controller for arm")
        from control.arm_controller import Pose
        base_pos = ARM_A_BASE if arm_label == "a" else ARM_B_BASE
        direction = np.asarray(target_pos, dtype=np.float64) - base_pos
        horizontal_angle = float(np.arctan2(direction[1], direction[0])) if np.linalg.norm(direction[:2]) > 1e-6 else 0.0

        grasp_quat = OBJECT_GRASP_ORIENTATIONS.get(object_name, (0.0, 0.0, 1.0, 0.0))
        if arm_label == "b":
            q = np.array(grasp_quat, dtype=np.float64)
            flip = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            grasp_quat = tuple(np.asarray([q[0]*flip[0] - q[1]*flip[1] - q[2]*flip[2] - q[3]*flip[3],
                                           q[0]*flip[1] + q[1]*flip[0] + q[2]*flip[3] - q[3]*flip[2],
                                           q[0]*flip[2] - q[1]*flip[3] + q[2]*flip[0] + q[3]*flip[1],
                                           q[0]*flip[3] + q[1]*flip[2] - q[2]*flip[1] + q[3]*flip[0]]))

        site_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(site_quat, controller.data.site_xmat[controller.model.site(controller.site_name).id])

        initial_guesses = [
            controller.joint_positions().copy(),
            np.array([horizontal_angle, 1.0, -0.8, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([horizontal_angle, 1.3, -1.0, 0.0, -0.3, 0.0], dtype=np.float64),
            np.array([horizontal_angle, 1.5, -0.5, 0.0, 0.5, 0.0], dtype=np.float64),
            np.array([horizontal_angle, 0.8, -1.2, 0.0, -0.5, 0.0], dtype=np.float64),
            np.array([horizontal_angle, 1.2, -0.6, 0.0, 0.0, 0.0], dtype=np.float64),
        ]
        best_joints = controller.joint_positions().copy()
        best_pos_error = float("inf")
        best_ori_error = float("inf")
        attempts = 0
        for guess in initial_guesses:
            guess = np.clip(guess, controller.joint_limits[:, 0], controller.joint_limits[:, 1])
            attempts += 1
            try:
                pose = Pose(tuple(np.asarray(target_pos, dtype=np.float64).tolist()), site_quat)
                joints = controller.inverse_kinematics(pose, max_iterations=IK_MAX_ITERATIONS, initial_guess=guess)
                fk = controller.forward_kinematics(joints)
                pos_error = float(np.linalg.norm(np.asarray(fk.position) - target_pos))
                ori_error = self._angular_error(np.asarray(fk.orientation), np.asarray(grasp_quat))
                if pos_error < best_pos_error:
                    best_pos_error = pos_error
                    best_ori_error = ori_error
                    best_joints = joints.copy()
                if pos_error < IK_POSITION_TOLERANCE and ori_error < IK_ORIENTATION_TOLERANCE:
                    LOGGER.debug("_ik_target arm=%s SUCCESS method=DIRECT pos_error=%.6f ori_error=%.6f attempts=%d",
                                  arm_label, pos_error, ori_error, attempts)
                    return IKResult(True, joints, pos_error, ori_error, "DIRECT", attempts)
            except Exception:
                continue

        for guess in initial_guesses:
            guess = np.clip(guess, controller.joint_limits[:, 0], controller.joint_limits[:, 1])
            attempts += 1
            try:
                pose = Pose(tuple(np.asarray(target_pos, dtype=np.float64).tolist()), grasp_quat)
                joints = controller.inverse_kinematics(pose, max_iterations=IK_MAX_ITERATIONS, initial_guess=guess)
                fk = controller.forward_kinematics(joints)
                pos_error = float(np.linalg.norm(np.asarray(fk.position) - target_pos))
                ori_error = self._angular_error(np.asarray(fk.orientation), np.asarray(grasp_quat))
                if pos_error < best_pos_error:
                    best_pos_error = pos_error
                    best_ori_error = ori_error
                    best_joints = joints.copy()
                if pos_error < IK_POSITION_TOLERANCE and ori_error < IK_ORIENTATION_TOLERANCE:
                    LOGGER.debug("_ik_target arm=%s SUCCESS method=DIRECT2 pos_error=%.6f ori_error=%.6f attempts=%d",
                                  arm_label, pos_error, ori_error, attempts)
                    return IKResult(True, joints, pos_error, ori_error, "DIRECT", attempts)
            except Exception:
                continue

        best_joints, best_dist = self._random_sampling_ik(controller, target_pos, initial_joints=best_joints)
        best_pos_error = best_dist
        best_ori_error = best_dist
        attempts += IK_MAX_RANDOM_SAMPLES
        if best_dist < IK_RANDOM_SAMPLE_THRESHOLD:
            try:
                pose = Pose(tuple(np.asarray(target_pos, dtype=np.float64).tolist()), grasp_quat)
                joints = controller.inverse_kinematics(pose, max_iterations=200, initial_guess=best_joints)
                fk = controller.forward_kinematics(joints)
                pos_error = float(np.linalg.norm(np.asarray(fk.position) - target_pos))
                ori_error = self._angular_error(np.asarray(fk.orientation), np.asarray(grasp_quat))
                if pos_error < IK_POSITION_TOLERANCE and ori_error < IK_ORIENTATION_TOLERANCE:
                    LOGGER.debug("_ik_target arm=%s SUCCESS method=RANDOM_REFINEMENT pos_error=%.6f ori_error=%.6f",
                                  arm_label, pos_error, ori_error)
                    return IKResult(True, joints, pos_error, ori_error, "RANDOM_REFINEMENT", attempts)
                if pos_error < best_pos_error:
                    best_pos_error = pos_error
                    best_ori_error = ori_error
                    best_joints = joints.copy()
            except Exception:
                pass

        LOGGER.warning("_ik_target arm=%s FAILED target=%s pos_error=%.4f ori_error=%.4f method=RANDOM attempts=%d",
                       arm_label, target_pos.tolist(), best_pos_error, best_ori_error, attempts)
        try:
            scipy_joints, scipy_pos_err = self._scipy_ik(controller, target_pos, grasp_quat, best_joints)
            if scipy_pos_err < best_pos_error:
                fk = controller.forward_kinematics(scipy_joints)
                pos_error = float(np.linalg.norm(np.asarray(fk.position) - target_pos))
                ori_error = self._angular_error(np.asarray(fk.orientation), np.asarray(grasp_quat))
                LOGGER.info("_ik_target arm=%s scipy fallback pos_error=%.4f ori_error=%.4f",
                            arm_label, pos_error, ori_error)
                if pos_error < IK_POSITION_TOLERANCE and ori_error < IK_ORIENTATION_TOLERANCE:
                    return IKResult(True, scipy_joints, pos_error, ori_error, "SCIPY", attempts, "IK converged via scipy")
                if pos_error < IK_POSITION_TOLERANCE * 3 and ori_error < 1.5:
                    return IKResult(True, scipy_joints, pos_error, ori_error, "SCIPY_PARTIAL", attempts,
                                   "IK converged position, orientation approximate")
                return IKResult(False, scipy_joints, pos_error, ori_error, "SCIPY", attempts,
                               "IK improved but did not converge")
        except Exception:
            pass
        return IKResult(False, best_joints, best_pos_error, best_ori_error, "RANDOM", attempts,
                        "IK did not converge within tolerance")

    def _scipy_ik(self, controller, target_pos, grasp_quat, initial_joints):
        """Solve IK using scipy optimization as fallback.

        Minimizes position error from current FK to target position.
        """
        import scipy.optimize

        target = np.asarray(target_pos, dtype=np.float64)
        q0 = np.asarray(initial_joints, dtype=np.float64)
        q0 = np.clip(q0, controller.joint_limits[:, 0], controller.joint_limits[:, 1])
        grasp_q = np.array(grasp_quat, dtype=np.float64)

        def objective(joints):
            joints = np.clip(joints, controller.joint_limits[:, 0], controller.joint_limits[:, 1])
            snapshot = controller._set_temporary_joints(joints)
            try:
                fk = controller.forward_kinematics(joints)
                pos_err = float(np.linalg.norm(np.asarray(fk.position) - target))
                ori_err = float(self._angular_error(np.asarray(fk.orientation), grasp_q))
            finally:
                controller._restore_temporary(snapshot)
            return pos_err + 0.1 * ori_err

        result = scipy.optimize.minimize(
            objective, q0, method="L-BFGS-B",
            bounds=[(controller.joint_limits[i, 0], controller.joint_limits[i, 1]) for i in range(6)],
            options={"maxiter": 500, "ftol": 1e-8}
        )
        best_joints = np.clip(result.x, controller.joint_limits[:, 0], controller.joint_limits[:, 1])
        return best_joints, float(result.fun)

    @staticmethod
    def _angular_error(current_quat: np.ndarray, target_quat: np.ndarray) -> float:
        """Compute angular error between two quaternions in radians."""
        dot = float(np.clip(np.dot(current_quat, target_quat), -1.0, 1.0))
        return 2.0 * np.arccos(abs(dot))

    def _random_sampling_ik(self, controller: Any, target_pos: np.ndarray, num_samples: int = IK_MAX_RANDOM_SAMPLES, initial_joints: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Random sampling IK fallback with local refinement.

        Returns the best joint configuration found and its position error.
        """
        best_joints = initial_joints.copy() if initial_joints is not None else controller.joint_positions().copy()
        best_dist = float("inf")
        target = np.asarray(target_pos, dtype=np.float64)
        base_pos = ARM_A_BASE if controller.label == "a" else ARM_B_BASE
        direction = target - base_pos
        horizontal_angle = float(np.arctan2(direction[1], direction[0])) if np.linalg.norm(direction[:2]) > 1e-6 else 0.0
        rng = np.random.default_rng(42)
        for _ in range(num_samples):
            joints = rng.uniform(controller.joint_limits[:, 0], controller.joint_limits[:, 1])
            if num_samples > 100 and _ < num_samples // 2:
                joints[0] = horizontal_angle + rng.uniform(-0.3, 0.3)
                joints[1] = rng.uniform(-1.2, 0.2)
            try:
                fk = controller.forward_kinematics(joints)
                dist = float(np.linalg.norm(np.asarray(fk.position) - target))
                if dist < best_dist:
                    best_dist = dist
                    best_joints = joints.copy()
                if dist < 0.01:
                    break
            except Exception:
                continue
        LOGGER.debug("_random_sampling_ik best_dist=%.4f", best_dist)
        return best_joints, best_dist

    def _command(self, name: str, action: Action | None = None) -> dict[str, Any]:
        """Convert semantic phases into safe deterministic joint commands."""
        objects = action.objects if action and action.objects else ()
        target = objects[0] if objects else ""
        arm = action.arm if action and action.arm else ("both" if len(objects) > 1 else None)
        arm_a_pose = SAFE_POSE.copy()
        arm_b_pose = SAFE_POSE.copy()
        gripper_a = 0.0
        gripper_b = 0.0
        drawer = None
        ik_failed = False

        def _ik_to_pose(ik_result: IKResult) -> np.ndarray:
            """Convert IKResult to joint pose array, logging warnings on failure."""
            if not ik_result.success:
                LOGGER.warning("_command IK failed for %s: %s", target, ik_result.failure_reason)
                nonlocal ik_failed
                ik_failed = True
            return ik_result.joints

        if name in {"observe", "locate_drawer", "locate_utensils", "re_observe", "re_plan", "correct_position", "verify", "finish", "hold", "locate"}:
            arm_a_pose = SAFE_POSE.copy()
            arm_b_pose = SAFE_POSE.copy()
            gripper_a = 0.0
            gripper_b = 0.0
        elif name == "open_drawer":
            ik_result = self._ik_target("a", DRAWER_HANDLE)
            arm_a_pose = _ik_to_pose(ik_result)
            gripper_a = 1.0
            drawer = -0.30
            self.metrics.left_arm_actions += 1
        elif name in {"approach", "pick", "retrieve", "pregrasp"}:
            arm_label = action.arm if action and action.arm else self._arm_for_object(target)
            arm_label = arm_label.lower() if isinstance(arm_label, str) else arm_label
            pos = self._object_position(target)
            LOGGER.info("DEBUG approach target=%s arm_label=%s pos=%s", target, arm_label, pos.tolist())
            offset = APPROACH_OFFSET.copy()
            if target in {"spoon", "fork"}:
                offset = np.array([0.0, 0.0, 0.04], dtype=np.float64)
            elif target in {"plate", "cup"}:
                offset = np.array([0.0, 0.0, 0.06], dtype=np.float64)
            ik_result = self._ik_target(arm_label, pos, offset, object_name=target)
            if arm_label == "a":
                arm_a_pose = _ik_to_pose(ik_result)
                gripper_a = 0.0
                self.metrics.left_arm_actions += 1
            else:
                arm_b_pose = _ik_to_pose(ik_result)
                gripper_b = 0.0
                self.metrics.right_arm_actions += 1
        elif name in {"grasp", "retry_grasp"}:
            arm_label = action.arm if action and action.arm else self._arm_for_object(target)
            arm_label = arm_label.lower() if isinstance(arm_label, str) else arm_label
            pos = self._object_position(target)
            ik_result = self._ik_target(arm_label, pos, object_name=target)
            if arm_label == "a":
                arm_a_pose = _ik_to_pose(ik_result)
                gripper_a = 1.0
                self.metrics.left_arm_actions += 1
            else:
                arm_b_pose = _ik_to_pose(ik_result)
                gripper_b = 1.0
                self.metrics.right_arm_actions += 1
            self._held_object = target
        elif name == "verify_grasp":
            arm_a_pose = SAFE_POSE.copy()
            arm_b_pose = SAFE_POSE.copy()
        elif name == "lift":
            if self._held_object:
                arm_label = action.arm if action and action.arm else self._arm_for_object(self._held_object)
                arm_label = arm_label.lower() if isinstance(arm_label, str) else arm_label
                lift_pos = self._object_position(self._held_object)
                lift_pos[2] += 0.15
                ik_result = self._ik_target(arm_label, lift_pos, object_name=self._held_object)
                if arm_label == "a":
                    arm_a_pose = _ik_to_pose(ik_result)
                    gripper_a = 1.0
                    self.metrics.left_arm_actions += 1
                else:
                    arm_b_pose = _ik_to_pose(ik_result)
                    gripper_b = 1.0
                    self.metrics.right_arm_actions += 1
            else:
                arm_a_pose = SAFE_POSE.copy()
                arm_b_pose = SAFE_POSE.copy()
        elif name == "transport":
            if self._held_object:
                arm_label = action.arm if action and action.arm else self._arm_for_object(self._held_object)
                arm_label = arm_label.lower() if isinstance(arm_label, str) else arm_label
                table_pos = TABLE_CENTER.copy()
                table_pos[0] += 0.15 if arm_label == "a" else -0.15
                table_pos[2] = TABLE_TOP + OBJECT_HALF_HEIGHTS.get(self._held_object, 0.02) + 0.1
                ik_result = self._ik_target(arm_label, table_pos, TRANSPORT_HEIGHT, object_name=self._held_object)
                if arm_label == "a":
                    arm_a_pose = _ik_to_pose(ik_result)
                    gripper_a = 1.0
                    self.metrics.left_arm_actions += 1
                else:
                    arm_b_pose = _ik_to_pose(ik_result)
                    gripper_b = 1.0
                    self.metrics.right_arm_actions += 1
            else:
                ik_failed = True
                arm_a_pose = SAFE_POSE.copy()
                arm_b_pose = SAFE_POSE.copy()
        elif name == "preplace":
            if self._held_object:
                arm_label = action.arm if action and action.arm else self._arm_for_object(self._held_object)
                arm_label = arm_label.lower() if isinstance(arm_label, str) else arm_label
                table_pos = TABLE_CENTER.copy()
                table_pos[0] += 0.15 if arm_label == "a" else -0.15
                table_pos[2] = TABLE_TOP + OBJECT_HALF_HEIGHTS.get(self._held_object, 0.02) + 0.05
                ik_result = self._ik_target(arm_label, table_pos, object_name=self._held_object)
                if arm_label == "a":
                    arm_a_pose = _ik_to_pose(ik_result)
                    gripper_a = 1.0
                    self.metrics.left_arm_actions += 1
                else:
                    arm_b_pose = _ik_to_pose(ik_result)
                    gripper_b = 1.0
                    self.metrics.right_arm_actions += 1
            else:
                ik_failed = True
                arm_a_pose = SAFE_POSE.copy()
                arm_b_pose = SAFE_POSE.copy()
        elif name in {"release", "place"}:
            if self._held_object:
                arm_label = action.arm if action and action.arm else self._arm_for_object(self._held_object)
                arm_label = arm_label.lower() if isinstance(arm_label, str) else arm_label
                table_pos = TABLE_CENTER.copy()
                table_pos[0] += 0.15 if arm_label == "a" else -0.15
                table_pos[2] = TABLE_TOP + OBJECT_HALF_HEIGHTS.get(self._held_object, 0.02)
                ik_result = self._ik_target(arm_label, table_pos, object_name=self._held_object)
                if arm_label == "a":
                    arm_a_pose = _ik_to_pose(ik_result)
                    gripper_a = 0.0
                    self.metrics.left_arm_actions += 1
                else:
                    arm_b_pose = _ik_to_pose(ik_result)
                    gripper_b = 0.0
                    self.metrics.right_arm_actions += 1
            else:
                ik_failed = True
                arm_a_pose = SAFE_POSE.copy()
                arm_b_pose = SAFE_POSE.copy()
                gripper_a = 0.0
                gripper_b = 0.0
        elif name == "verify_placement":
            arm_a_pose = SAFE_POSE.copy()
            arm_b_pose = SAFE_POSE.copy()
        elif name == "retract":
            arm_a_pose = SAFE_POSE.copy()
            arm_b_pose = SAFE_POSE.copy()
            gripper_a = 0.0
            gripper_b = 0.0
        elif name in {"coordinate", "synchronized_reach", "handoff"}:
            arm_a_pose = SAFE_POSE.copy()
            arm_b_pose = SAFE_POSE.copy()
            gripper_a = 1.0
            gripper_b = 1.0
            self.metrics.handoffs += int(name == "handoff")
        elif name == "fail":
            arm_a_pose = SAFE_POSE.copy()
            arm_b_pose = SAFE_POSE.copy()
            gripper_a = 0.0
            gripper_b = 0.0

        result = {
            "arm_a": np.asarray(arm_a_pose, dtype=np.float64),
            "arm_b": np.asarray(arm_b_pose, dtype=np.float64),
            "gripper_a": float(np.clip(gripper_a, 0.0, 1.0)),
            "gripper_b": float(np.clip(gripper_b, 0.0, 1.0)),
            "drawer": drawer,
            "semantic_action": name,
        }
        if ik_failed:
            result["ik_failed"] = True
            LOGGER.warning("_command IK FAILED for %s: SAFE_POSE used as recovery, not success", target)
        if arm == "both" or (action and action.arm == "both"):
            self.metrics.bimanual_actions += 1
        elif name not in {"coordinate", "synchronized_reach", "handoff"}:
            self.metrics.single_arm_actions += 1
        return result


def diagnose_scene(model: Any, data: Any) -> str:
    """Output detailed scene geometry for debugging."""
    lines = ["OBJECT SCENE DEBUG"]
    geom_names = {
        "plate": ["plate_geom"],
        "cup": ["cup_geom"],
        "spoon": ["spoon_handle", "spoon_bowl"],
        "fork": ["fork_handle", "fork_head"],
    }
    for name, names in geom_names.items():
        try:
            body_id = model.body(name).id
            pos = data.xpos[body_id]
            quat = data.xquat[body_id]
            size = None
            for gname in names:
                try:
                    geom_id = model.geom(gname).id
                    size = model.geom_size[geom_id]
                    break
                except (KeyError, ValueError):
                    continue
            size_str = f"radius={size[0]}, half_height={size[1]}" if size is not None else "unknown"
            lines.append(f"{name}:")
            lines.append(f"  body_name={name}")
            lines.append(f"  position={pos.tolist()}")
            lines.append(f"  quaternion={quat.tolist()}")
            lines.append(f"  dimensions={size_str}")
        except Exception:
            lines.append(f"{name}: NOT FOUND")
    for label in ("a", "b"):
        try:
            base_body = model.body(f"so101_{label}_base").id
            base_pos = data.xpos[base_body]
            wrist_body = model.body(f"arm_{label}_wrist").id
            wrist_pos = data.xpos[wrist_body]
            ee_body = model.body(f"arm_{label}_gripper_left").id
            ee_pos = data.xpos[ee_body]
            lines.append(f"arm_{label}:")
            lines.append(f"  base={base_pos.tolist()}")
            lines.append(f"  ee_position={ee_pos.tolist()}")
            lines.append(f"  wrist_position={wrist_pos.tolist()}")
        except Exception:
            lines.append(f"arm_{label}: NOT FOUND")
    return "\n".join(lines)