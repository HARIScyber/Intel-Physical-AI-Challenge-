"""Pure Python reward tests."""
import unittest
from simulation.rewards import placement_reward

class RewardTests(unittest.TestCase):
    def test_zero_distance_is_success_reward(self) -> None:
        self.assertEqual(placement_reward(0.0), 1.0)

if __name__ == "__main__":
    unittest.main()
