"""Tests for the deterministic scripted baseline policy."""

import unittest
import numpy as np

from planning.action_sequence import Action, ActionSequence
from policy import ScriptedPolicy


class ScriptedPolicyTests(unittest.TestCase):
    """Verify semantic phases become deterministic robot commands."""

    def setUp(self) -> None:
        self.sequence = ActionSequence((
            Action("open_drawer", objects=("drawer",)),
            Action("approach", objects=("spoon",)),
            Action("grasp", objects=("spoon",)),
            Action("transport", objects=("spoon",)),
            Action("release", objects=("spoon",)),
            Action("retract"),
            Action("coordinate", arm="both"),
            Action("handoff", objects=("spoon",), arm="both"),
            Action("verify"),
        ), "baseline")
        self.policy = ScriptedPolicy()
        self.policy.reset(self.sequence)

    def test_phase_commands_are_deterministic(self) -> None:
        commands = [self.policy.predict(object(), self.sequence) for _ in self.sequence.actions]
        self.assertEqual(commands[0]["drawer"], -0.30)
        self.assertEqual(commands[1]["gripper_a"], 0.0)
        self.assertEqual(commands[2]["gripper_a"], 1.0)
        self.assertEqual(commands[4]["gripper_a"], 0.0)
        self.assertEqual(commands[6]["gripper_b"], 1.0)
        self.assertEqual(commands[7]["semantic_action"], "handoff")
        self.assertEqual(self.policy.metrics.handoffs, 1)

    def test_metrics_finalize_from_scene_state(self) -> None:
        scene = type("Scene", (), {"robot_state": {"task_state": {"complete": True}}})()
        metrics = self.policy.complete(scene, 12, 1.5, 0)
        self.assertTrue(metrics["task_success"])
        self.assertEqual(metrics["step_count"], 12)

    def test_object_positions_updated_from_scene(self) -> None:
        class FakeObj:
            def __init__(self, name, position):
                self.name = name
                self.position = position
        class FakeScene:
            def __init__(self):
                self.detected_objects = [FakeObj("spoon", (0.1, -0.18, 0.86))]
        scene = FakeScene()
        cmd = self.policy.predict(scene, self.sequence)
        self.assertIsInstance(cmd["arm_a"], np.ndarray)

    def test_bimanual_metrics_counted(self) -> None:
        seq = ActionSequence((
            Action("approach", objects=("spoon",), arm="a"),
            Action("approach", objects=("cup",), arm="b"),
            Action("coordinate", arm="both"),
        ), "bimanual")
        policy = ScriptedPolicy()
        policy.reset(seq)
        for _ in seq.actions:
            policy.predict(object(), seq)
        self.assertEqual(policy.metrics.left_arm_actions, 1)
        self.assertEqual(policy.metrics.right_arm_actions, 1)
        self.assertEqual(policy.metrics.bimanual_actions, 1)


if __name__ == "__main__":
    unittest.main()