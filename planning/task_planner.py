"""Hierarchical, model-independent task planner for dinner-table manipulation."""
from __future__ import annotations
from typing import Any
from language.command_schema import TaskCommand
from perception.scene_state import SceneState
from .action_sequence import Action, ActionSequence
from .bimanual_planner import BimanualPlanner
from .state_machine import PlanState, PlanStateMachine

class TaskPlanner:
    """Expand semantic commands into recoverable executable action sequences."""
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.state_machine = PlanStateMachine(int(self.config.get("max_retries", 2)))
        self.bimanual = BimanualPlanner(self.config)

    def plan(self, command: TaskCommand, scene: SceneState) -> ActionSequence:
        """Create a hierarchical plan from a language command and scene state."""
        if not command.goal or scene is None:
            raise ValueError("command and scene are required")
        self.state_machine.reset()
        self.state_machine.transition(PlanState.OBSERVE)
        actions = [Action("observe", 0.2)]
        missing = (set(command.objects) - {"table"}) - set(scene.objects)
        if missing:
            actions.append(Action("re_observe", 0.5, tuple(sorted(missing)), recovery_for="missing_object"))
        self.state_machine.transition(PlanState.PLAN)
        actions.extend(self._set_table_actions(command, scene) if command.goal == "set_dinner_table" else self._command_actions(command))
        if command.requires_bimanual:
            actions.append(Action("coordinate", 0.5, arm="both", parameters={"collision_check": True}))
        actions.extend((Action("verify", 0.5, parameters={"retry_on_failure": True}), Action("finish", 0.1)))
        return self.bimanual.coordinate(ActionSequence(tuple(actions), command.goal))

    def recovery_actions(self, failure_type: str, objects: tuple[str, ...] = ()) -> ActionSequence:
        """Return a bounded recovery subplan for an execution failure."""
        state = self.state_machine.recovery_state(failure_type)
        if state == PlanState.FAILED:
            return ActionSequence((Action("fail", 0.1, recovery_for=failure_type),), "failed")
        if failure_type == "missing_object":
            actions = (Action("re_observe", 0.5, objects, recovery_for=failure_type),)
        elif failure_type == "grasp_failed":
            actions = (Action("re_observe", 0.3, objects, recovery_for=failure_type), Action("retry_grasp", 0.7, objects, recovery_for=failure_type))
        elif failure_type == "collision_risk":
            actions = (Action("re_plan", 0.5, objects, recovery_for=failure_type, parameters={"increase_clearance": True}),)
        else:
            actions = (Action("correct_position", 0.7, objects, recovery_for=failure_type), Action("verify", 0.5, objects, recovery_for=failure_type))
        return ActionSequence(actions, "recovery")

    def _set_table_actions(self, command: TaskCommand, scene: SceneState) -> list[Action]:
        """Build the explicit dinner-table hierarchy."""
        actions = [Action("locate_drawer", 0.3, ("drawer",))]
        if not bool(scene.drawer_state.get("open", False)):
            actions.append(Action("open_drawer", 0.8, ("drawer",), arm="a"))
        actions.append(Action("locate_utensils", 0.3, ("spoon", "fork")))
        for item in ("spoon", "fork", "plate", "cup"):
            if item in command.objects:
                from policy.base_policy import OBJECT_ARM_MAP
                arm = OBJECT_ARM_MAP.get(item, "a")
                actions.extend((
                    Action("approach", 0.4, (item,), arm=arm),
                    Action("grasp", 0.5, (item,), arm=arm),
                    Action("transport", 0.6, (item,), arm=arm),
                    Action("release", 0.3, (item,), arm=arm),
                    Action("retract", 0.3, arm=arm),
                ))
        return actions

    @staticmethod
    def _command_actions(command: TaskCommand) -> list[Action]:
        """Translate normalized command steps into executable actions."""
        return [Action(step.action, 0.8, step.objects, step.arm, step.relation, {"quantity": step.quantity}) for step in command.steps]

def plan_task(command: TaskCommand, scene: SceneState | None = None, config: dict[str, Any] | None = None) -> ActionSequence:
    """Create a plan, using an empty scene for backward compatibility."""
    return TaskPlanner(config).plan(command, scene or SceneState())