"""Unit tests for the MuJoCo-only dual-arm control layer."""

import unittest

import numpy as np

from control import BimanualController
from physical_ai.config import load_config
from simulation import BimanualMujocoEnv


class DualArmControllerTests(unittest.TestCase):
    """Exercise controller contracts against the compiled simulation model."""

    def setUp(self) -> None:
        self.environment = BimanualMujocoEnv(load_config())
        self.environment.reset(seed=31)
        self.controller = BimanualController(self.environment)

    def tearDown(self) -> None:
        self.environment.close()

    def test_separate_arm_state_and_fk(self) -> None:
        state = self.controller.get_state()
        self.assertEqual(state["arm_a"]["name"], "arm_a")
        self.assertEqual(state["arm_b"]["name"], "arm_b")
        self.assertEqual(len(state["arm_a"]["joint_positions"]), 6)
        self.assertNotEqual(state["arm_a"]["end_effector_pose"], state["arm_b"]["end_effector_pose"])

    def test_ik_round_trip_at_current_pose(self) -> None:
        pose = self.controller.arm_a.forward_kinematics()
        joints = self.controller.arm_a.inverse_kinematics(pose)
        np.testing.assert_allclose(joints, self.controller.arm_a.joint_positions(), atol=1e-3)

    def test_limits_and_velocity_are_checked(self) -> None:
        current = self.controller.arm_a.joint_positions()
        with self.assertRaises(ValueError):
            self.controller.arm_a.move_joint([10.0] * 6)
        with self.assertRaises(ValueError):
            self.controller.arm_a.move_joint(current + 0.5, duration=0.01)

    def test_gripper_commands_and_atomic_move(self) -> None:
        self.controller.arm_a.open_gripper()
        self.assertGreater(self.controller.arm_a.gripper_state()["opening"], 0.9)
        self.controller.arm_a.close_gripper()
        self.assertLess(self.controller.arm_a.gripper_state()["opening"], 0.1)
        result = self.controller.move_both(
            self.controller.arm_a.joint_positions(),
            self.controller.arm_b.joint_positions(),
            execute=False,
        )
        self.assertIn("arm_a", result)
        self.assertIn("arm_b", result)

    def test_invalid_parallel_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.controller.parallel_actions({"arm_a": {"joint_targets": [0.0] * 6}})


if __name__ == "__main__":
    unittest.main()