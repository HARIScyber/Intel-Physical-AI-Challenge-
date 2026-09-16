"""Tests for optional model-independent VLA policy integration."""

import unittest
from pathlib import Path

from policy import DummyPolicy, LeRobotVLA, ScriptedPolicy, create_policy


class VLAPolicyTests(unittest.TestCase):
    """Verify optional loading never breaks dependency-free operation."""

    def test_factory_defaults_to_scripted_without_checkpoint(self) -> None:
        self.assertIsInstance(create_policy({"type": "smolvla", "checkpoint": None}), ScriptedPolicy)
        self.assertIsInstance(create_policy({"type": "scripted"}), ScriptedPolicy)
        self.assertIsInstance(create_policy({"type": "dummy"}), DummyPolicy)

    def test_dummy_output_matches_dual_arm_interface(self) -> None:
        action = DummyPolicy().predict({"camera_images": {}, "robot_joint_positions": []}, "set the table")
        self.assertEqual(action["arm_a"].shape, (6,))
        self.assertEqual(action["arm_b"].shape, (6,))

    def test_vla_requires_checkpoint_before_inference(self) -> None:
        with self.assertRaises(Exception):
            LeRobotVLA({"type": "smolvla"}).predict({}, "set the table")

    def test_vla_states_enum(self) -> None:
        from policy.vla_policy import VLAStates
        self.assertEqual(VLAStates.UNINITIALIZED.value, "uninitialized")
        self.assertEqual(VLAStates.READY.value, "ready")
        self.assertEqual(VLAStates.FAILED.value, "failed")
        self.assertEqual(VLAStates.UNAVAILABLE.value, "unavailable")
        self.assertEqual(VLAStates.FALLBACK.value, "fallback")

    def test_vla_initialization(self) -> None:
        vla = LeRobotVLA()
        from policy.vla_policy import VLAStates
        self.assertEqual(vla.state, VLAStates.UNINITIALIZED)
        self.assertFalse(vla.is_ready())

    def test_vla_load_checkpoint_failure(self) -> None:
        vla = LeRobotVLA()
        with self.assertRaises(Exception):
            vla.load_checkpoint("nonexistent_local_checkpoint_that_does_not_exist")
        from policy.vla_policy import VLAStates
        self.assertEqual(vla.state, VLAStates.FAILED)

    def test_vla_predict_without_checkpoint(self) -> None:
        vla = LeRobotVLA()
        with self.assertRaises(Exception):
            vla.predict({}, "set the table")


if __name__ == "__main__":
    unittest.main()