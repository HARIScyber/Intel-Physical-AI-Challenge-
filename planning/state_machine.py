"""Recoverable execution state machine for planned manipulation tasks."""
from __future__ import annotations
import logging
from enum import Enum
LOGGER = logging.getLogger(__name__)

class PlanState(str, Enum):
    """Lifecycle states required by the task planner."""
    IDLE = "idle"
    OBSERVE = "observe"
    PLAN = "plan"
    MOVE = "move"
    GRASP = "grasp"
    TRANSPORT = "transport"
    PLACE = "place"
    HANDOFF = "handoff"
    VERIFY = "verify"
    RECOVER = "recover"
    COMPLETE = "complete"
    FAILED = "failed"

class PlanStateMachine:
    """Track plan execution and choose bounded recovery transitions."""
    def __init__(self, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.state = PlanState.IDLE
        self.max_retries = max_retries
        self.retry_count = 0
        self.last_error: str | None = None

    def transition(self, state: PlanState) -> PlanState:
        """Move to a valid execution state."""
        if not isinstance(state, PlanState):
            raise ValueError("state must be a PlanState")
        self.state = state
        LOGGER.debug("Planner state -> %s", state.value)
        return state

    def recover(self, failure: str) -> PlanState:
        """Enter recovery and retry, or fail after the configured retry budget."""
        if not failure:
            raise ValueError("failure description is required")
        self.last_error = failure
        if self.retry_count >= self.max_retries:
            self.state = PlanState.FAILED
        else:
            self.retry_count += 1
            self.state = PlanState.RECOVER
        return self.state

    def recovery_state(self, failure_type: str) -> PlanState:
        """Map known failure classes to bounded recovery."""
        if failure_type not in {"missing_object", "grasp_failed", "collision_risk", "placement_failed"}:
            raise ValueError(f"unsupported failure type: {failure_type}")
        return self.recover(failure_type)

    def reset(self) -> None:
        """Reset the machine for a new task."""
        self.state, self.retry_count, self.last_error = PlanState.IDLE, 0, None