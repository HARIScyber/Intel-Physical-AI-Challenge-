"""Executable action schemas for hierarchical and bimanual planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """One executable action with explicit coordination and safety contracts."""

    name: str
    duration: float = 1.0
    objects: tuple[str, ...] = ()
    arm: str | None = None
    relation: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    recovery_for: str | None = None
    target: str | None = None
    priority: int = 0
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable action mapping."""
        return asdict(self)


@dataclass(frozen=True)
class ActionSequence:
    """Ordered executable actions with an optional task goal."""

    actions: tuple[Action, ...]
    goal: str = ""

    def __post_init__(self) -> None:
        """Reject empty or invalid action sequences early."""
        if not self.actions:
            raise ValueError("action sequence cannot be empty")
        if any(action.duration <= 0 for action in self.actions):
            raise ValueError("action durations must be positive")

    def names(self) -> tuple[str, ...]:
        """Return action names for logging and assertions."""
        return tuple(action.name for action in self.actions)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable sequence mapping."""
        return {"goal": self.goal, "actions": [action.as_dict() for action in self.actions]}