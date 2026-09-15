"""Model-independent VLA policy facade with safe optional loading."""

from __future__ import annotations

from enum import Enum
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
        
        Args:
            checkpoint: Optional checkpoint path. If None, uses config.
            
        Raises:
            VLAUnavailableError: If checkpoint loading fails
        """
        self.state = VLAStates.LOADING
        try:
            self.adapter.load_checkpoint(checkpoint)
            self.loaded_checkpoint = checkpoint
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

    def predict(self, scene_state: Any, action_sequence: str | ActionSequence | None = None) -> dict[str, Any]:
        """Return a dual-arm action from camera/state/language input.
        
        Args:
            scene_state: Current scene/state observation
            action_sequence: Optional action sequence to follow
            
        Returns:
            Command action dictionary or None if prediction fails
            
        Raises:
            VLAUnavailableError: If VLA is not ready for inference
        """
        if not self.is_ready():
            raise VLAUnavailableError(f"VLA policy is not ready (state: {self.state}). Call load_checkpoint() first.")
        
        try:
            return self.adapter.predict(scene_state, action_sequence)
        except LeRobotUnavailableError as exc:
            self.state = VLAStates.UNAVAILABLE
            raise VLAUnavailableError(str(exc)) from exc

    def close(self) -> None:
        """Clean up VLA resources."""
        self.state = VLAStates.UNINITIALIZED
        self.adapter.reset()