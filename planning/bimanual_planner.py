"""Shared-workspace, collision-aware planning for two simulated SO-101 arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .action_sequence import Action, ActionSequence
from .collision_checker import CollisionChecker


@dataclass(frozen=True)
class WorkspaceBounds:
    """World-space bounds shared by both arms."""

    minimum: tuple[float, float, float] = (-1.0, -0.65, 0.70)
    maximum: tuple[float, float, float] = (1.0, 0.65, 1.65)
    min_arm_clearance: float = 0.12

    def contains(self, position: Iterable[float]) -> bool:
        """Return whether a 3D position lies inside the shared workspace."""
        point = np.asarray(tuple(position), dtype=float)
        return point.shape == (3,) and bool(np.all(point >= self.minimum) and np.all(point <= self.maximum))


class BimanualPlanner:
    """Plan complementary arm actions with safety checks before every action."""

    def __init__(self, config: dict[str, Any] | None = None, model: Any = None, data: Any = None) -> None:
        self.config = config or {}
        self.model = model
        self.data = data
        self.workspace = WorkspaceBounds(
            tuple(self.config.get("workspace_min", (-1.0, -0.65, 0.70))),
            tuple(self.config.get("workspace_max", (1.0, 0.65, 1.65))),
            float(self.config.get("min_arm_clearance", 0.12)),
        )
        self.collision_checker = CollisionChecker(model, data, self.workspace.min_arm_clearance)
        self.arm_priority = {"A": 1, "B": 1, "BOTH": 2}

    def coordinate(self, sequence: ActionSequence) -> ActionSequence:
        """Normalize arm labels and insert synchronized reach before BOTH actions."""
        actions: list[Action] = []
        for action in sequence.actions:
            normalized = self._normalize_action(action)
            if normalized.arm == "BOTH":
                actions.append(self._action("synchronized_reach", arm="BOTH", priority=normalized.priority, preconditions=("both_arms_available", "collision_free"), postconditions=("arms_reached",)))
            actions.append(normalized)
        return ActionSequence(tuple(actions), sequence.goal)

    def plan_parallel(self, actions: Iterable[Action]) -> ActionSequence:
        """Create a complementary parallel sequence, resolving priority conflicts."""
        normalized = [self._normalize_action(action) for action in actions]
        if not normalized:
            raise ValueError("parallel plan cannot be empty")
        arms = {action.arm for action in normalized}
        if arms != {"A", "B"}:
            raise ValueError("parallel plan requires one A action and one B action")
        if any(action.priority == other.priority and action.target == other.target for action in normalized for other in normalized if action is not other):
            normalized.sort(key=lambda action: (action.priority, action.arm or ""))
        return ActionSequence(tuple(normalized), "parallel")

    def handoff(self, object_name: str, from_arm: str = "A", to_arm: str = "B") -> ActionSequence:
        """Build a guarded source-to-receiver handoff sequence."""
        source, receiver = self._arm_label(from_arm), self._arm_label(to_arm)
        if source == receiver:
            raise ValueError("handoff requires distinct arms")
        return ActionSequence((
            self._action("reach", source, (object_name,), object_name, 2, ("object_available", "workspace_safe"), ("source_reached",)),
            self._action("grasp", source, (object_name,), object_name, 2, ("source_reached", "gripper_open"), ("object_grasped",)),
            self._action("receive", receiver, (object_name,), object_name, 2, ("receiver_available", "collision_free"), ("receiver_ready",)),
            self._action("handoff", "BOTH", (object_name,), object_name, 3, ("object_grasped", "receiver_ready"), ("object_transferred",)),
            self._action("release", source, (object_name,), object_name, 2, ("object_transferred",), ("source_released",)),
        ), "handoff")

    def cooperative_manipulation(self, object_name: str, target: str) -> ActionSequence:
        """Build synchronized complementary grasp, transport, and placement actions."""
        return ActionSequence((
            self._action("cooperative_grasp", "BOTH", (object_name,), target, 3, ("object_available", "both_grippers_open", "collision_free"), ("object_cooperatively_grasped",)),
            self._action("cooperative_transport", "BOTH", (object_name,), target, 3, ("object_cooperatively_grasped", "workspace_safe"), ("object_at_target",)),
            self._action("cooperative_place", "BOTH", (object_name,), target, 3, ("object_at_target",), ("object_placed",)),
        ), "cooperative_manipulation")

    def validate_action(self, action: Action, scene: Any = None, joint_targets: dict[str, Any] | None = None) -> tuple[bool, tuple[str, ...]]:
        """Check workspace, collision, availability, grippers, and target before execution."""
        failures: list[str] = []
        normalized = self._normalize_action(action)
        if scene is not None:
            available = set(getattr(scene, "objects", {}))
            required = set(normalized.objects) - {"table", "drawer"}
            if required - available:
                failures.append("object_unavailable")
            grippers = getattr(scene, "robot_state", {}).get("grippers", {})
            if "gripper_open" in normalized.preconditions and grippers and not all(grippers.values()):
                failures.append("gripper_not_open")
        if normalized.target and scene is not None:
            targets = getattr(scene, "table_state", {}).get("targets", {})
            if normalized.target in targets and not self.workspace.contains(targets[normalized.target]):
                failures.append("target_outside_workspace")
        if joint_targets is not None and self.model is not None:
            if normalized.arm == "BOTH":
                clear = self.collision_checker.is_collision_free([joint_targets["A"], joint_targets["B"]], ("a", "b"))
            else:
                clear = self.collision_checker.is_collision_free([joint_targets[normalized.arm]], (normalized.arm.lower(),))
            if not clear:
                failures.append("collision_risk")
        return not failures, tuple(failures)

    def replan(self, sequence: ActionSequence, failed_action: Action, failure_reasons: Iterable[str]) -> ActionSequence:
        """Replan after a safety failure by inserting observe/replan actions."""
        reasons = tuple(failure_reasons)
        if not reasons:
            raise ValueError("failure reasons are required")
        actions = list(sequence.actions)
        try:
            index = actions.index(failed_action)
        except ValueError:
            index = 0
        recovery = self._action("re_observe", failed_action.arm, failed_action.objects, failed_action.target, failed_action.priority, ("scene_updated",), ("replanned",), recovery_for=", ".join(reasons))
        actions[index:index + 1] = [recovery, self._action("re_plan", failed_action.arm, failed_action.objects, failed_action.target, failed_action.priority, ("replanned",), failed_action.postconditions, recovery_for=", ".join(reasons))]
        return ActionSequence(tuple(actions), sequence.goal)

    def _normalize_action(self, action: Action) -> Action:
        """Normalize legacy arm names to the public A/B/BOTH representation."""
        arm = {"arm_a": "A", "arm_b": "B", "both": "BOTH", "a": "A", "b": "B"}.get(action.arm, action.arm)
        if arm not in {None, "A", "B", "BOTH"}:
            raise ValueError(f"unsupported arm label: {action.arm}")
        return Action(action.name, action.duration, action.objects, arm, action.relation, action.parameters, action.recovery_for, action.target, action.priority or self.arm_priority.get(arm or "A", 1), action.preconditions, action.postconditions)

    def _action(self, name: str, arm: str | None, objects: tuple[str, ...] = (), target: str | None = None, priority: int = 1, preconditions: tuple[str, ...] = (), postconditions: tuple[str, ...] = (), recovery_for: str | None = None) -> Action:
        """Construct an explicit bimanual action record."""
        return Action(name, 0.5, objects, arm, None, {"safety_checks": ("workspace", "collision", "availability", "gripper", "target")}, recovery_for, target, priority, preconditions, postconditions)

    @staticmethod
    def _arm_label(value: str) -> str:
        """Normalize a caller arm label."""
        label = {"arm_a": "A", "arm_b": "B", "a": "A", "b": "B"}.get(value.lower(), value.upper())
        if label not in {"A", "B"}:
            raise ValueError("arm must be A or B")
        return label


def coordinate(sequence: ActionSequence, config: dict | None = None) -> ActionSequence:
    """Backward-compatible coordination entry point."""
    return BimanualPlanner(config).coordinate(sequence)