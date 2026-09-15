"""MuJoCo ground-truth perception tests."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from physical_ai.config import load_config
from perception import CameraProcessor
from simulation import BimanualMujocoEnv


class PerceptionTests(unittest.TestCase):
    """Verify multimodal state and debug-image contracts."""

    def test_ground_truth_scene_state_contains_configured_objects(self) -> None:
        environment = BimanualMujocoEnv(load_config())
        try:
            observation, _ = environment.reset(seed=41)
            with TemporaryDirectory() as output_dir:
                processor = CameraProcessor(load_config().simulation["perception"], output_dir)
                scene = processor.perceive(observation, environment.model, environment.data)
                self.assertEqual({item.name for item in scene.detected_objects}, {"plate", "cup", "spoon", "fork", "napkin", "bowl"})
                self.assertTrue(all(item.confidence == 1.0 for item in scene.detected_objects))
                self.assertTrue(all(item.bounding_box[2] > item.bounding_box[0] for item in scene.detected_objects))
                self.assertTrue((Path(output_dir) / "overhead_rgb.png").is_file())
                self.assertTrue((Path(output_dir) / "overhead_detected_objects.png").is_file())
                self.assertTrue((Path(output_dir) / "overhead_segmentation.png").is_file())
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()