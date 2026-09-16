"""Isolated adapter for the installed LeRobot policy factory API."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .base_policy import BasePolicy

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


class LeRobotUnavailableError(RuntimeError):
    """Raised when LeRobot or a compatible Python runtime is unavailable."""


class LeRobotPolicy(BasePolicy):
    """Generic LeRobot policy adapter using documented factory methods."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.policy: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.loaded_checkpoint: str | None = None

    def load_checkpoint(self, checkpoint: str | None = None) -> None:
        """Load a LeRobot policy through ``make_policy_config`` and ``make_policy``."""
        reference = checkpoint or self.config.get("checkpoint")
        if not reference:
            raise LeRobotUnavailableError("No VLA checkpoint configured")
        reference_str = str(reference)
        source_root = ROOT / "lerobot" / "src"
        if source_root.is_dir() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            from lerobot.policies import make_policy, make_policy_config, make_pre_post_processors
            from lerobot.common.utils.hub import hub_available
        except (ImportError, ModuleNotFoundError, SyntaxError) as exc:
            raise LeRobotUnavailableError(f"LeRobot is not available: {exc}") from exc
        reference_path = Path(reference_str)
        is_local = reference_path.exists() and reference_path.is_dir()
        if is_local:
            LOGGER.info("VLA CHECKPOINT VALIDATION checkpoint=%s checkpoint_type=local exists=true", reference_str)
        else:
            LOGGER.info("VLA CHECKPOINT VALIDATION checkpoint=%s checkpoint_type=%s exists=%s", reference_str, "local" if not hub_available() else "hub_or_local", is_local)
            if not hub_available() and not is_local:
                raise LeRobotUnavailableError(
                    f"VLA CHECKPOINT ERROR requested_checkpoint={reference_str} checkpoint_type=local exists=false "
                    "reason=checkpoint path does not exist and HuggingFace hub is unavailable "
                    "action=VLA disabled; use scripted fallback or provide a valid checkpoint"
                )
        policy_type = str(self.config.get("type", "smolvla"))
        device = str(self.config.get("device", "cpu"))
        try:
            policy_config = make_policy_config(policy_type, pretrained_path=reference_str, device=device)
            self.policy = make_policy(policy_config, env_cfg=self.config.get("env_config"))
            preprocessor_overrides = {"device_processor": {"device": device}}
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_config, pretrained_path=reference_str, preprocessor_overrides=preprocessor_overrides
            )
            self.loaded_checkpoint = reference_str
        except (ImportError, ModuleNotFoundError, SyntaxError, ValueError, RuntimeError, OSError) as exc:
            raise LeRobotUnavailableError(f"Unable to load LeRobot policy '{reference_str}': {exc}") from exc

    def reset(self) -> None:
        """Clear LeRobot action queues for a new episode."""
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, observation: Any, instruction: str | None = None) -> Any:
        """Preprocess camera/state/language input and select one LeRobot action."""
        if self.policy is None:
            raise LeRobotUnavailableError("Call load_checkpoint() before predict()")
        if not isinstance(observation, dict) or not instruction:
            raise ValueError("observation mapping and instruction are required")
        batch = _to_lerobot_batch(observation, instruction)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        return _to_dual_arm_action(action)

    def predict_with_action_sequence(self, observation: Any, action_sequence: Any, instruction: str | None = None) -> Any:
        """Preprocess and predict using action sequence."""
        return self.predict(observation, instruction)

    def predict_with_action_sequence(self, observation: Any, action_sequence: Any, instruction: str | None = None) -> Any:
        """Preprocess and predict using action sequence."""
        return self.predict(observation, instruction)


def _to_lerobot_batch(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Map project observation keys to standard LeRobot observation keys."""
    import torch
    batch: dict[str, Any] = {
        "observation.state": torch.as_tensor(np.asarray(observation["robot_joint_positions"], dtype=np.float32)),
        "task": instruction,
    }
    for camera_name, image in observation.get("camera_images", {}).items():
        batch[f"observation.images.{camera_name}"] = torch.as_tensor(np.asarray(image)).permute(2, 0, 1)
    return batch


def _to_dual_arm_action(action: Any) -> dict[str, np.ndarray | float]:
    """Convert a LeRobot action tensor into the project dual-arm interface."""
    values = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
    values = values.reshape(-1)
    if values.size < 12:
        raise ValueError(f"LeRobot action has {values.size} values; at least 12 joint values are required")
    return {
        "arm_a": values[:6].astype(np.float64),
        "arm_b": values[6:12].astype(np.float64),
        "gripper_a": float(values[12]) if values.size > 12 else 0.0,
        "gripper_b": float(values[13]) if values.size > 13 else 0.0,
    }
