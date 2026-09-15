"""Domain randomization and robustness tests."""

import unittest

import numpy as np

from evaluation.robustness import evaluate_modes, replay_seed
from physical_ai.config import load_config
from simulation import BimanualMujocoEnv
from simulation.randomization import language_variant, mode_scale, randomize_control_noise


class RandomizationTests(unittest.TestCase):
    """Verify modes, seeds, records, and replay behavior."""

    def test_modes_have_ordered_difficulty(self) -> None:
        self.assertLess(mode_scale("easy"), mode_scale("medium"))
        self.assertLess(mode_scale("medium"), mode_scale("hard"))

    def test_reset_is_reproducible_and_records_seed(self) -> None:
        environment = BimanualMujocoEnv(load_config())
        try:
            first, info_first = environment.reset(seed=101, options={"evaluation_mode": "hard"})
            first_record = info_first["randomization"]
            second, info_second = environment.reset(seed=101, options={"evaluation_mode": "hard"})
            self.assertEqual(info_first["seed"], 101)
            self.assertEqual(first_record, info_second["randomization"])
            np.testing.assert_allclose(first["robot_joint_positions"], second["robot_joint_positions"])
        finally:
            environment.close()

    def test_noise_and_language_are_seeded(self) -> None:
        first = randomize_control_noise(np.random.default_rng(4), np.zeros(3), {"randomization": {"control_noise": 0.1}}, "hard")
        second = randomize_control_noise(np.random.default_rng(4), np.zeros(3), {"randomization": {"control_noise": 0.1}}, "hard")
        np.testing.assert_allclose(first, second)
        self.assertEqual(language_variant(np.random.default_rng(8), "hard"), language_variant(np.random.default_rng(8), "hard"))

    def test_robustness_replay(self) -> None:
        trial = lambda seed, mode: seed == 7 and mode == "hard"
        results = evaluate_modes(trial, [7, 8], ("easy", "hard"))
        self.assertEqual(len(results), 4)
        self.assertTrue(replay_seed(trial, 7, "hard").success)


if __name__ == "__main__":
    unittest.main()