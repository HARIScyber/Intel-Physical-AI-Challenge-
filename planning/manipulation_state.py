"""Object manipulation state machine for tracking grasp/transport/place progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ManipulationState(str, Enum):
    """States in the object manipulation lifecycle."""

    NOT_STARTED = "not_started"
    LOCATED = "located"
    PREGRASP = "pregrasp"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    GRASPED = "grasped"
    LIFTING = "lifting"
    TRANSPORTING = "transporting"
    PREPLACING = "preplacing"
    RELEASING = "releasing"
    PLACED = "placed"
    VERIFIED = "verified"
    FAILED = "failed"
    RECOVERY = "recovery"


# Valid state transitions
VALID_TRANSITIONS = {
    ManipulationState.NOT_STARTED: {ManipulationState.LOCATED, ManipulationState.FAILED},
    ManipulationState.LOCATED: {ManipulationState.PREGRASP, ManipulationState.FAILED},
    ManipulationState.PREGRASP: {ManipulationState.APPROACHING, ManipulationState.FAILED},
    ManipulationState.APPROACHING: {ManipulationState.GRASPING, ManipulationState.FAILED},
    ManipulationState.GRASPING: {ManipulationState.GRASPED, ManipulationState.FAILED},
    ManipulationState.GRASPED: {ManipulationState.LIFTING, ManipulationState.FAILED},
    ManipulationState.LIFTING: {ManipulationState.TRANSPORTING, ManipulationState.FAILED},
    ManipulationState.TRANSPORTING: {ManipulationState.PREPLACING, ManipulationState.FAILED},
    ManipulationState.PREPLACING: {ManipulationState.RELEASING, ManipulationState.FAILED},
    ManipulationState.RELEASING: {ManipulationState.PLACED, ManipulationState.FAILED},
    ManipulationState.PLACED: {ManipulationState.VERIFIED, ManipulationState.FAILED},
    ManipulationState.VERIFIED: {ManipulationState.FAILED},
    ManipulationState.FAILED: {ManipulationState.RECOVERY},
    ManipulationState.RECOVERY: {ManipulationState.NOT_STARTED, ManipulationState.FAILED},
}


@dataclass
class ObjectManipulationState:
    """Tracks the manipulation state of a single object."""

    name: str
    state: ManipulationState = ManipulationState.NOT_STARTED
    assigned_arm: str | None = None
    grasp_verified: bool = False
    placement_verified: bool = False
    failure_reason: str = ""
    retry_count: int = 0
    max_retries: int = 2

    def transition(self, new_state: ManipulationState, reason: str = "") -> bool:
        """Attempt to transition to a new state.

        Returns True if transition is valid, False otherwise.
        """
        if new_state not in VALID_TRANSITIONS.get(self.state, set()):
            self.failure_reason = f"Invalid transition from {self.state} to {new_state}: {reason}"
            return False

        self.state = new_state
        if new_state == ManipulationState.FAILED:
            self.failure_reason = reason
        return True

    def can_proceed_to(self, next_state: ManipulationState) -> bool:
        """Check if a transition to next_state is valid."""
        return next_state in VALID_TRANSITIONS.get(self.state, set())

    def is_terminal(self) -> bool:
        """Return True if in a terminal state (VERIFIED or FAILED after retries)."""
        if self.state == ManipulationState.VERIFIED:
            return True
        if self.state == ManipulationState.FAILED and self.retry_count >= self.max_retries:
            return True
        return False

    def record_failure(self, reason: str) -> None:
        """Record a failure and increment retry count."""
        self.state = ManipulationState.FAILED
        self.failure_reason = reason
        self.retry_count += 1


class ManipulationStateMachine:
    """Manages manipulation state machines for all objects in a task."""

    def __init__(self, required_objects: tuple[str, ...], max_retries: int = 2) -> None:
        self.objects: dict[str, ObjectManipulationState] = {
            name: ObjectManipulationState(name=name, max_retries=max_retries)
            for name in required_objects
        }
        self._global_state = ManipulationState.NOT_STARTED

    def get_state(self, object_name: str) -> ManipulationState:
        """Get current state of an object."""
        return self.objects.get(object_name, ObjectManipulationState(name=object_name)).state

    def transition(self, object_name: str, new_state: ManipulationState, reason: str = "") -> bool:
        """Transition an object to a new state."""
        if object_name not in self.objects:
            return False
        return self.objects[object_name].transition(new_state, reason)

    def record_failure(self, object_name: str, reason: str) -> bool:
        """Record a failure for an object."""
        if object_name not in self.objects:
            return False
        self.objects[object_name].record_failure(reason)
        return True

    def can_transport(self, object_name: str) -> bool:
        """Check if object can be transported (must be GRASPED)."""
        obj = self.objects.get(object_name)
        return obj is not None and obj.state == ManipulationState.GRASPED

    def can_release(self, object_name: str) -> bool:
        """Check if object can be released (must be TRANSPORTING or PREPLACING)."""
        obj = self.objects.get(object_name)
        return obj is not None and obj.state in {ManipulationState.TRANSPORTING, ManipulationState.PREPLACING}

    def all_verified(self) -> bool:
        """Check if all objects are VERIFIED."""
        return all(obj.state == ManipulationState.VERIFIED for obj in self.objects.values())

    def any_failed_terminal(self) -> bool:
        """Check if any object has failed terminally."""
        return any(obj.is_terminal() and obj.state == ManipulationState.FAILED for obj in self.objects.values())

    def get_history(self) -> dict[str, set[str]]:
        """Get manipulation history for TaskVerifier."""
        history = {}
        for name, obj in self.objects.items():
            states = set()
            # Add all states up to and including current
            state_order = [
                ManipulationState.LOCATED,
                ManipulationState.PREGRASP,
                ManipulationState.APPROACHING,
                ManipulationState.GRASPING,
                ManipulationState.GRASPED,
                ManipulationState.LIFTING,
                ManipulationState.TRANSPORTING,
                ManipulationState.PREPLACING,
                ManipulationState.RELEASING,
                ManipulationState.PLACED,
                ManipulationState.VERIFIED,
            ]
            current_index = state_order.index(obj.state) if obj.state in state_order else -1
            for i in range(current_index + 1):
                states.add(state_order[i].value)
            if obj.grasp_verified:
                states.add("grasp_verified")
            if obj.placement_verified:
                states.add("placement_verified")
            history[name] = states
        return history

    def set_arm(self, object_name: str, arm: str) -> None:
        """Assign an arm to an object."""
        if object_name in self.objects:
            self.objects[object_name].assigned_arm = arm

    def get_arm(self, object_name: str) -> str | None:
        """Get assigned arm for an object."""
        obj = self.objects.get(object_name)
        return obj.assigned_arm if obj else None


def create_state_machine(task_command: Any, max_retries: int = 2) -> ManipulationStateMachine:
    """Create a manipulation state machine from a task command."""
    if task_command and task_command.objects:
        required = tuple(obj for obj in task_command.objects if obj not in {"table", "drawer"})
        if required:
            return ManipulationStateMachine(required, max_retries)
    # Default dinner table objects
    return ManipulationStateMachine(("plate", "cup", "spoon", "fork"), max_retries)