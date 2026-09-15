"""Advanced bimanual planner tests."""

import unittest

from perception.scene_state import DetectedObject, SceneState
from planning import Action, ActionSequence, BimanualPlanner


class BimanualPlannerTests(unittest.TestCase):
    """Verify shared-workspace and recovery behavior without hardware."""

    def setUp(self) -> None:
        self.planner = BimanualPlanner()
        self.scene = SceneState(
            detected_objects=[DetectedObject("spoon", "spoon", (0.0, 0.0, 0.85), (1, 1, 5, 5), 1.0, "overhead")],
            table_state={"targets": {"receiving_position": (0.2, 0.1, 0.85)}},
            robot_state={"grippers": {"A": True, "B": True}},
        )

    def test_action_schema_contains_coordination_fields(self) -> None:
        sequence = self.planner.handoff("spoon")
        action = sequence.actions[3]
        self.assertEqual(action.arm, "BOTH")
        self.assertEqual(action.target, "spoon")
        self.assertGreater(action.priority, 0)
        self.assertIn("object_grasped", action.preconditions)
        self.assertIn("object_transferred", action.postconditions)

    def test_workspace_and_scene_safety_checks(self) -> None:
        safe = Action("place", objects=("spoon",), arm="A", target="receiving_position", preconditions=("gripper_open",))
        valid, reasons = self.planner.validate_action(safe, self.scene)
        self.assertTrue(valid)
        self.assertEqual(reasons, ())
        missing = Action("pick", objects=("fork",), arm="A")
        valid, reasons = self.planner.validate_action(missing, self.scene)
        self.assertFalse(valid)
        self.assertIn("object_unavailable", reasons)

    def test_parallel_and_cooperative_plans(self) -> None:
        parallel = self.planner.plan_parallel((Action("open_drawer", arm="A", priority=2), Action("prepare_table", arm="B", priority=1)))
        self.assertEqual({action.arm for action in parallel.actions}, {"A", "B"})
        cooperative = self.planner.cooperative_manipulation("plate", "table_center")
        self.assertEqual(cooperative.names(), ("cooperative_grasp", "cooperative_transport", "cooperative_place"))
        self.assertTrue(all(action.arm == "BOTH" for action in cooperative.actions))

    def test_dynamic_replanning_after_collision(self) -> None:
        failed = Action("reach", arm="A", objects=("spoon",), target="table_center")
        original = ActionSequence((failed, Action("place", arm="A", objects=("spoon",))), "test")
        replanned = self.planner.replan(original, failed, ("collision_risk",))
        self.assertEqual(replanned.names()[:2], ("re_observe", "re_plan"))
        self.assertEqual(replanned.actions[0].recovery_for, "collision_risk")

    def test_both_action_gets_synchronized_reach(self) -> None:
        sequence = ActionSequence((Action("cooperate", arm="both"),), "test")
        coordinated = self.planner.coordinate(sequence)
        self.assertEqual(coordinated.names(), ("synchronized_reach", "cooperate"))


if __name__ == "__main__":
    unittest.main()