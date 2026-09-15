"""Tests for demonstration dataset validation."""

import unittest
from pathlib import Path

from project_datasets.validate_dataset import validate


class DatasetTests(unittest.TestCase):
    """Validate the generated local episode when present."""

    def test_raw_dataset_is_valid(self) -> None:
        dataset = Path(__file__).resolve().parents[2] / "project_datasets" / "raw"
        if not dataset.exists():
            self.skipTest("local demonstrations have not been generated")
        self.assertTrue(validate(dataset))


if __name__ == "__main__":
    unittest.main()