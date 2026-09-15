"""Validation and deterministic task reasoning for dinner-table commands."""

from __future__ import annotations

from typing import Any

from .command_schema import TaskCommand, TaskStep


class TaskReasoner:
    """Validate parser output and add safe execution constraints."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def reason(self, command: TaskCommand) -> TaskCommand:
        """Return a validated command suitable for task planning."""
        if not command.goal or not command.steps:
            raise ValueError("command must contain a goal and at least one step")
        allowed = {"plate", "cup", "spoon", "fork", "napkin", "bowl", "drawer", "table"}
        if any(item not in allowed for item in command.objects):
            raise ValueError("command contains an unsupported object")
        constraints = list(command.constraints)
        if command.requires_bimanual and "avoid_inter_arm_collision" not in constraints:
            constraints.append("avoid_inter_arm_collision")
        steps = tuple(self._normalize_step(step, command.requires_bimanual) for step in command.steps)
        return TaskCommand(command.goal, command.objects, tuple(constraints), command.requires_bimanual, steps, command.raw_instruction, command.confidence)

    @staticmethod
    def _normalize_step(step: TaskStep, bimanual: bool) -> TaskStep:
        """Assign a preferred arm hint while keeping execution collision-safe."""
        if step.quantity < 1:
            raise ValueError("step quantity must be positive")
        return TaskStep(step.action, step.objects, step.relation, step.quantity, step.arm or ("both" if bimanual else None))


def reason_about_task(command: TaskCommand, config: dict[str, Any] | None = None) -> TaskCommand:
    """Backward-compatible functional reasoning entry point."""
    return TaskReasoner(config).reason(command)