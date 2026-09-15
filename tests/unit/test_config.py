"""Configuration smoke tests."""
import unittest
from pathlib import Path

from physical_ai.config import load_config

class ConfigTests(unittest.TestCase):
    def test_loads_repository_configuration(self) -> None:
        config = load_config()
        self.assertEqual(config.project_root, Path(__file__).resolve().parents[2])
        self.assertEqual(config.simulation["simulation"]["seed"], 7)

if __name__ == "__main__":
    unittest.main()
