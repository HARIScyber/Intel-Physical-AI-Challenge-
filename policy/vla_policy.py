"""Model-independent VLA policy facade with safe optional loading."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy, LeRobotUnavailableError
from planning.action_sequence import ActionSequence


class VLAStates(Enum):
    """VLA policy lifecycle states."""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    FALLBACK = "fallback"


class VLAUnavailableError(LeRobotUnavailableError):
    """Raised when a configured VLA cannot be loaded or inferred."""


class LeRobotVLA(BasePolicy):
    """SmolVLA-capable facade isolated from the rest of the robotics stack."""

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None) -> None:
        """Initialize VLA policy with optional device configuration.
        
        Args:
            config: Optional configuration dictionary
            device: Target device (cpu/gpu)
        """
        super().__init__(config)
        self.device = device or "cpu"
        self.state = VLAStates.UNINITIALIZED
        self.adapter = LeRobotPolicy(config)
        self.loaded_checkpoint: str | None = None

    def is_ready(self) -> bool:
        """Check if the VLA policy is ready for inference.
        
        Returns:
            True if VLA is in READY state, False otherwise
        """
        return self.state == VLAStates.READY

    def load_checkpoint(self, checkpoint: str | None = None) -> None:
        """Load the configured LeRobot VLA checkpoint lazily.

        Validates local checkpoint existence before loading.
        Does not treat nonexistent paths as HuggingFace model IDs.

        Raises:
            VLAUnavailableError: If checkpoint loading fails or path doesn't exist locally
        """
        self.state = VLAStates.LOADING
        reference = checkpoint or self.config.get("checkpoint")
        if reference is None:
            self.state = VLAStates.FAILED
            raise VLAUnavailableError("No VLA checkpoint configured")
        reference_path = Path(str(reference))
        if not reference_path.exists():
            self.state = VLAStates.FAILED
            raise VLAUnavailableError(
                f"VLA CHECKPOINT ERROR requested_checkpoint={reference} "
                "checkpoint_type=local exists=false "
                "action=VLA disabled; use scripted fallback or provide a valid checkpoint"
            )
        try:
            self.adapter.load_checkpoint(reference)
            self.loaded_checkpoint = reference
            self.state = VLAStates.READY
        except Exception as exc:
            self.state = VLAStates.FAILED
            if isinstance(exc, LeRobotUnavailableError):
                raise VLAUnavailableError(str(exc)) from exc
            raise VLAUnavailableError(str(exc)) from exc

    def reset(self, action_sequence: ActionSequence | None = None) -> None:
        """Reset model action queues and optional action sequence.
        
        Args:
            action_sequence: Optional sequence to initialize policy with
        """
        self.adapter.reset()

    def predict(self, scene_state: Any, action_sequence: str | ActionSequence | None = None, instruction: str | None = None) -> dict[str, Any]:
        """Return a dual-arm action from camera/state/language input.
        
        Args:
            scene_state: Current scene/state observation
            action_sequence: Optional action sequence to follow
            instruction: Natural-language instruction for VLA model
            
        Returns:
            Command action dictionary or None if prediction fails
            
        Raises:
            VLAUnavailableError: If VLA is not ready for inference
        """
        if not self.is_ready():
            raise VLAUnavailableError(f"VLA policy is not ready (state: {self.state}). Call load_checkpoint() first.")
        
        try:
            return self.adapter.predict(scene_state, instruction)
        except LeRobotUnavailableError as exc:
            self.state = VLAStates.UNAVAILABLE
            raise VLAUnavailableError(str(exc)) from exc

    def predict_with_action_sequence(self, scene_state: Any, action_sequence: Any, instruction: str | None = None) -> dict[str, Any]:
        """Preprocess and predict using action sequence."""
        return self.predict(scene_state, action_sequence, instruction)

    def close(self) -> None:
        """Clean up VLA resources."""
        self.state = VLAStates.UNINITIALIZED
        self.adapter.reset()