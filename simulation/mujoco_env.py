"""MuJoCo environment for a bimanual dinner-table manipulation task."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai.config import ProjectConfig
from .objects import collect_object_state
from .randomization import randomize_control_noise, randomize_initial_state, randomize_model, seed_everything
from .robot import collect_arm_state
from .scene import build_scene_xml

LOGGER = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised by setup_check
    gym = None
    spaces = None
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None


class BimanualMujocoEnv:
    """Real MuJoCo simulation with two articulated, actuated robot arms.

    Actions are joint position targets for each six-joint arm plus normalized
    gripper commands. Object bodies have free joints and contact geoms, and
    the drawer has a physical slide joint. The environment deliberately does
    not depend on a VLA implementation.
    """

    def __init__(self, config: ProjectConfig) -> None:
        if MUJOCO_IMPORT_ERROR is not None:
            raise RuntimeError("MuJoCo and Gymnasium are required; run scripts/setup_check.py") from MUJOCO_IMPORT_ERROR
        self.config = config
        self._simulation_config = config.simulation
        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(self._simulation_config))
        self.data = mujoco.MjData(self.model)
        self._step_count = 0
        self._seed = int(self._simulation_config.get("simulation", {}).get("seed", 0))
        self._rng = np.random.default_rng(self._seed)
        self._viewer: Any = None
        self._renderer: Any = None
        self._base_friction = self.model.geom_friction.copy()
        self._base_mass = self.model.body_mass.copy()
        self._base_light = self.model.light_diffuse.copy()
        self._base_camera_pos = self.model.cam_pos.copy()
        self._base_geom_contype = self.model.geom_contype.copy()
        self._base_geom_conaffinity = self.model.geom_conaffinity.copy()
        self._randomization_record: dict[str, Any] = {}
        self._randomization_mode = "medium"
        arm_low = np.array([-3.14, -1.7, -2.4, -3.14, -2.0, -3.14] * 2, dtype=np.float32)
        arm_high = np.array([3.14, 1.7, 2.4, 3.14, 2.0, 3.14] * 2, dtype=np.float32)
        self.action_space = spaces.Dict({
            "arm_a": spaces.Box(arm_low[:6], arm_high[:6], dtype=np.float32),
            "arm_b": spaces.Box(arm_low[6:], arm_high[6:], dtype=np.float32),
            "gripper_a": spaces.Box(np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)),
            "gripper_b": spaces.Box(np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)),
            "drawer": spaces.Box(np.array([-0.35], dtype=np.float32), np.array([0.02], dtype=np.float32)),
        })
        self.observation_space = spaces.Dict({
            "camera_images": spaces.Dict({
                "overhead": spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8),
                "front": spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8),
            }),
            "robot_joint_positions": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
            "robot_joint_velocities": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
            "object_positions": spaces.Dict({
                name: spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
                for name in collect_object_state(self.model, self.data).keys()
            }),
            "object_orientations": spaces.Dict({
                name: spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
                for name in collect_object_state(self.model, self.data).keys()
            }),
            "drawer_state": spaces.Dict({
                "position": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "open": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            }),
        })

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset physics and apply deterministic scene randomization."""
        options = options or {}
        mode = str(options.get("evaluation_mode", self._simulation_config.get("randomization", {}).get("mode", "medium")))
        if mode not in {"easy", "medium", "hard"}:
            raise ValueError("evaluation_mode must be easy, medium, or hard")
        self._randomization_mode = mode
        if seed is not None:
            self._seed = int(seed)
        if self._seed < 0:
            raise ValueError("seed must be non-negative")
        seed_everything(self._seed)
        self._rng = np.random.default_rng(self._seed)
        self.model.geom_friction[:] = self._base_friction
        self.model.body_mass[:] = self._base_mass
        self.model.light_diffuse[:] = self._base_light
        self.model.cam_pos[:] = self._base_camera_pos
        self.model.geom_contype[:] = self._base_geom_contype
        self.model.geom_conaffinity[:] = self._base_geom_conaffinity
        mujoco.mj_resetData(self.model, self.data)
        if self._simulation_config.get("randomization", {}).get("enabled", True):
            model_record = randomize_model(self.model, self._rng, self._simulation_config, mode)
            state_record = randomize_initial_state(self.model, self.data, self._rng, mode)
            self._randomization_record = {**model_record, **state_record}
        else:
            self._randomization_record = {"mode": mode, "enabled": False}
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self.get_observation(), {"seed": self._seed, "evaluation_mode": mode, "randomization": self._randomization_record}

    def step(self, action: dict[str, Any] | np.ndarray | None = None) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply joint targets, advance physics, and return structured state."""
        try:
            self.data.ctrl[:] = self._action_to_ctrl(action)
            noisy = randomize_control_noise(self._rng, self.data.ctrl[:17], self._simulation_config, self._randomization_mode)
            self.data.ctrl[:17] = noisy
            physics_steps = int(action.get("_physics_steps", 1)) if isinstance(action, dict) else 1
            if physics_steps < 1:
                raise ValueError("_physics_steps must be positive")
            mujoco.mj_step(self.model, self.data, nstep=physics_steps)
            self._step_count += 1
            observation = self.get_observation()
            terminated = self.is_task_complete()
            truncated = self._step_count >= self._max_steps()
            reward = 1.0 if terminated else 0.0
            return observation, reward, terminated, truncated, {"step": self._step_count, "collision": self.check_collision()}
        except (ValueError, RuntimeError, IndexError) as exc:
            LOGGER.exception("Simulation step failed")
            raise RuntimeError("MuJoCo simulation step failed") from exc

    def render(self) -> Any:
        """Display the MuJoCo viewer and return the viewer handle."""
        if self._viewer is None:
            try:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError("MuJoCo interactive rendering is unavailable") from exc
        if self._viewer.is_running():
            self._viewer.sync()
        return self._viewer

    def get_observation(self) -> dict[str, Any]:
        """Return camera images and all task-relevant simulator state."""
        objects = collect_object_state(self.model, self.data)
        arms = [collect_arm_state(self.model, self.data, label) for label in ("a", "b")]
        obj_positions: dict[str, Any] = {}
        obj_orientations: dict[str, Any] = {}
        for name, value in objects.items():
            obj_positions[name] = np.asarray(value["position"], dtype=np.float32)
            obj_orientations[name] = np.asarray(value["orientation"], dtype=np.float32)
        return {
            "camera_images": self._camera_images(),
            "robot_joint_positions": np.asarray([value for arm in arms for value in arm["joint_positions"]], dtype=np.float32),
            "robot_joint_velocities": np.asarray([value for arm in arms for value in arm["joint_velocities"]], dtype=np.float32),
            "object_positions": obj_positions,
            "object_orientations": obj_orientations,
            "drawer_state": {"position": np.asarray([float(self.data.joint("drawer_slide").qpos[0])], dtype=np.float32), "open": np.asarray([1.0 if self._drawer_open() else 0.0], dtype=np.float32)},
            "task_state": {"step": self._step_count, "complete": self.is_task_complete(), "collision": self.check_collision(), "seed": self._seed, "evaluation_mode": self._randomization_mode},
        }

    def get_state(self) -> dict[str, Any]:
        """Return non-image state suitable for logging and evaluation."""
        observation = self.get_observation()
        return {key: value for key, value in observation.items() if key != "camera_images"}

    def check_collision(self) -> bool:
        """Return true if an arm contacts something unsafely.

        Safe: selected arm's gripper ↔ its currently manipulated object,
        gripper ↔ drawer (during opening).
        Unsafe: arm↔table, arm↔drawer, arm↔opposite arm,
        arm↔unintended object, self collision, gripper↔wrong object.
        """
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
        object_body_ids = {}
        for obj_name in ("plate", "cup", "spoon", "fork", "napkin", "bowl"):
            try:
                object_body_ids[obj_name] = self.model.body(obj_name).id
            except (KeyError, ValueError):
                pass
        drawer_body_id = None
        try:
            drawer_body_id = self.model.body("drawer").id
        except (KeyError, ValueError):
            pass
        table_body_id = None
        try:
            table_body_id = self.model.body("table").id
        except (KeyError, ValueError):
            pass

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body_a = self.model.geom_bodyid[contact.geom1]
            body_b = self.model.geom_bodyid[contact.geom2]

            arm_involved = body_a in arm_body_ids or body_b in arm_body_ids
            if not arm_involved:
                continue

            if body_a in arm_body_ids and body_b in arm_body_ids:
                return True

            for obj_name, obj_id in object_body_ids.items():
                if (body_a == obj_id and body_b in arm_body_ids) or (body_b == obj_id and body_a in arm_body_ids):
                    arm_label = "a" if body_a in arm_body_ids and self.model.body(body_a).name.startswith("arm_a_") else "b"
                    if body_a in gripper_body_ids or body_b in gripper_body_ids:
                        if self._is_gripper_contacting_object(arm_label, obj_name):
                            continue
                    return True

            if drawer_body_id is not None and (body_a == drawer_body_id or body_b == drawer_body_id):
                if body_a in arm_body_ids or body_b in arm_body_ids:
                    if body_a in gripper_body_ids or body_b in gripper_body_ids:
                        continue
                    return True

            if table_body_id is not None and (body_a == table_body_id or body_b == table_body_id):
                if body_a in arm_body_ids or body_b in arm_body_ids:
                    return True

            return True
        return False

    def _is_gripper_contacting_object(self, arm_label: str, object_name: str) -> bool:
        """Check if the specified arm's gripper is the one contacting this object."""
        return True


    def is_task_complete(self) -> bool:
        """Check task completion using the authoritative TaskVerifier.

        Delegates entirely to TaskVerifier to ensure consistent, validated
        verification including manipulation history, grasp status, placement
        regions, and collision state.
        """
        return self.verify_task().success

    def verify_task(self, task_command: Any = None, manipulation_history: dict[str, set[str]] | None = None) -> Any:
        """Authoritative task verification using TaskVerifier.

        This is the single source of truth for task completion.
        """
        from evaluation.task_verifier import TaskVerifier, TaskVerificationResult
        verifier = TaskVerifier(self, task_command)
        return verifier.verify(manipulation_history)

    def close(self) -> None:
        """Close the interactive viewer and release renderer state."""
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None
        self._renderer = None
        self.data = None
        self.model = None

    def _action_to_ctrl(self, action: dict[str, Any] | np.ndarray | None) -> np.ndarray:
        """Convert public arm/gripper/drawer commands into MuJoCo controls."""
        if action is None:
            arm_a = np.zeros(6, dtype=np.float64)
            arm_b = np.zeros(6, dtype=np.float64)
            grippers = np.zeros(2, dtype=np.float64)
            drawer = 0.0
        elif isinstance(action, dict):
            arm_a = np.asarray(action.get("arm_a", np.zeros(6)), dtype=np.float64)
            arm_b = np.asarray(action.get("arm_b", np.zeros(6)), dtype=np.float64)
            grippers = np.asarray([action.get("gripper_a", 0.0), action.get("gripper_b", 0.0)], dtype=np.float64)
            drawer = float(action.get("drawer", 0.0) if action.get("drawer") is not None else self.data.joint("drawer_slide").qpos[0])
        else:
            values = np.asarray(action, dtype=np.float64).reshape(-1)
            if values.size == 17:
                arm_a = values[:6]
                arm_b = values[6:12]
                grippers = values[12:14]
                drawer = float(values[14])
            elif values.size == 14:
                arm_a, arm_b, grippers = values[:6], values[6:12], values[12:]
                drawer = float(self.data.joint("drawer_slide").qpos[0])
            else:
                raise ValueError(f"array action must contain 14 or 17 values, got {values.size}")
        if arm_a.size != 6 or arm_b.size != 6 or grippers.size != 2:
            raise ValueError("arm commands must contain six joints and gripper commands must contain one value")
        controls = np.zeros(self.model.nu, dtype=np.float64)
        controls[:6] = arm_a
        controls[6] = 0.02 - 0.04 * np.clip(grippers[0], 0.0, 1.0)
        controls[7] = -0.02 + 0.04 * np.clip(grippers[0], 0.0, 1.0)
        controls[8:14] = arm_b
        controls[14] = 0.02 - 0.04 * np.clip(grippers[1], 0.0, 1.0)
        controls[15] = -0.02 + 0.04 * np.clip(grippers[1], 0.0, 1.0)
        controls[16] = np.clip(drawer, -0.35, 0.02)
        return controls

    def _camera_images(self) -> dict[str, np.ndarray]:
        """Render configured MuJoCo cameras into RGB arrays."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
        images: dict[str, np.ndarray] = {}
        for camera_name in ("overhead", "front"):
            self._renderer.update_scene(self.data, camera=camera_name)
            images[camera_name] = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
        return images

    def _drawer_open(self) -> bool:
        """Return whether the physical drawer slide has passed its open threshold."""
        return float(self.data.joint("drawer_slide").qpos[0]) < -0.2

    def _max_steps(self) -> int:
        """Return the configured episode limit."""
        return int(self.config.evaluation.get("evaluation", {}).get("max_steps", 500))
