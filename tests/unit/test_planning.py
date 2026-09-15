"""Hierarchical planner and recovery tests."""

import unittest

from language import parse_instruction
from perception.scene_state import DetectedObject, SceneState
from planning import PlanState, PlanStateMachine, TaskPlanner


def scene_with(*names: str, drawer_open: bool = False) -> SceneState:
    """Build a coordinate-free scene fixture from named detections."""
    objects = [DetectedObject(name, name, (0.0, 0.0, 0.8), (1, 1, 4, 4), 1.0, "overhead") for name in names]
    return SceneState(objects, {"open": drawer_open}, {"body": "table"}, {"joint_positions": []})


class PlanningTests(unittest.TestCase):
    """Verify hierarchical expansion and bounded recovery."""

    def test_set_table_sequence(self) -> None:
        command = parse_instruction("Set the dinner table.")
        sequence = TaskPlanner().plan(command, scene_with("plate", "cup", "spoon", "fork"))
        names = sequence.names()
        self.assertEqual(names[0], "observe")
        self.assertIn("locate_drawer", names)
        self.assertIn("open_drawer", names)
        self.assertIn("approach", names)
        self.assertIn("grasp", names)
        self.assertIn("transport", names)
        self.assertIn("release", names)
        self.assertIn("retract", names)
        self.assertEqual(names[-2:], ("verify", "finish"))

    def test_missing_object_reobserves(self) -> None:
        sequence = TaskPlanner().plan(parse_instruction("Set the dinner table."), scene_with("plate"))
        self.assertEqual(sequence.actions[1].name, "re_observe")
        self.assertEqual(sequence.actions[1].recovery_for, "missing_object")

    def test_all_states_and_recovery(self) -> None:
        self.assertEqual({state.value for state in PlanState}, {"idle", "observe", "plan", "move", "grasp", "transport", "place", "handoff", "verify", "recover", "complete", "failed"})
        machine = PlanStateMachine(max_retries=1)
        self.assertEqual(machine.recovery_state("grasp_failed"), PlanState.RECOVER)
        self.assertEqual(machine.recovery_state("grasp_failed"), PlanState.FAILED)

    def test_recovery_actions_cover_failure_modes(self) -> None:
        planner = TaskPlanner({"max_retries": 4})
        self.assertEqual(planner.recovery_actions("missing_object").names(), ("re_observe",))
        self.assertEqual(planner.recovery_actions("grasp_failed").names(), ("re_observe", "retry_grasp"))
        self.assertEqual(planner.recovery_actions("collision_risk").names(), ("re_plan",))
        self.assertEqual(planner.recovery_actions("placement_failed").names(), ("correct_position", "verify"))

    def test_bimanual_plan_gets_synchronized_reach(self) -> None:
        command = parse_instruction("Use both arms to organize the table.")
        sequence = TaskPlanner().plan(command, scene_with("plate", "cup", "spoon", "fork"))
        self.assertIn("synchronized_reach", sequence.names())


if __name__ == "__main__":
    unittest.main()