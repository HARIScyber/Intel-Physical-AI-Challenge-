"""Natural-language command understanding tests."""

import unittest

from language import InstructionParser, TaskReasoner


class LanguageTests(unittest.TestCase):
    """Verify deterministic dinner-table commands and safe reasoning."""

    def setUp(self) -> None:
        self.parser = InstructionParser()

    def test_set_table(self) -> None:
        command = self.parser.parse("Set the dinner table.")
        self.assertEqual(command.goal, "set_dinner_table")
        self.assertEqual(command.objects, ("plate", "cup", "spoon", "fork"))
        self.assertGreaterEqual(len(command.steps), 4)

    def test_drawer_quantity(self) -> None:
        command = self.parser.parse("Open the drawer and take out two spoons.")
        self.assertEqual(command.goal, "retrieve_from_drawer")
        self.assertEqual(command.steps[-1].quantity, 2)
        self.assertEqual(command.steps[0].action, "open_drawer")

    def test_relation_and_bimanual_requirement(self) -> None:
        adjacent = self.parser.parse("Put the cup next to the plate.")
        self.assertEqual(adjacent.goal, "place_next_to")
        self.assertIn("preserve_relative_adjacency", adjacent.constraints)
        both = TaskReasoner().reason(self.parser.parse("Use both arms to organize the table."))
        self.assertTrue(both.requires_bimanual)
        self.assertIn("avoid_inter_arm_collision", both.constraints)
        self.assertEqual(both.steps[0].arm, "both")

    def test_transformers_backend_falls_back_without_model(self) -> None:
        command = InstructionParser({"backend": "transformers", "model_name": "", "local_files_only": True}).parse("Bring me a spoon.")
        self.assertEqual(command.goal, "retrieve_object")

    def test_invalid_command(self) -> None:
        with self.assertRaises(ValueError):
            self.parser.parse("")
        with self.assertRaises(ValueError):
            self.parser.parse("Put the cup next to the moon.")


if __name__ == "__main__":
    unittest.main()