"""Deterministic command parsing with an optional local Transformers backend."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .command_schema import LanguageConfig, TaskCommand, TaskStep

LOGGER = logging.getLogger(__name__)
OBJECT_ALIASES = {"plates": "plate", "cups": "cup", "spoons": "spoon", "forks": "fork", "napkins": "napkin", "bowls": "bowl", "dishes": "plate"}
OBJECTS = ("plate", "cup", "spoon", "fork", "napkin", "bowl", "drawer", "table")


class InstructionParser:
    """Parse instructions using rules or a configured local Transformers model."""

    def __init__(self, config: dict[str, Any] | LanguageConfig | None = None) -> None:
        self.config = config if isinstance(config, LanguageConfig) else LanguageConfig.from_mapping(config)
        self._pipeline: Any = None

    def parse(self, instruction: str) -> TaskCommand:
        """Convert natural language into a structured command."""
        if not instruction or not instruction.strip():
            raise ValueError("instruction must be non-empty")
        if self.config.backend == "transformers":
            try:
                return self._parse_transformers(instruction.strip())
            except (ImportError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                LOGGER.warning("Transformers parser unavailable; using rules: %s", exc)
        return self._parse_rules(instruction.strip())

    def _parse_rules(self, instruction: str) -> TaskCommand:
        """Parse supported dinner-table language without external dependencies."""
        text = instruction.lower().strip()
        objects = self._objects_in(text)
        bimanual = bool(re.search(r"both arms|two arms|bimanual|with both", text))
        constraints: list[str] = []
        if "set the dinner table" in text or "set dinner table" in text:
            goal = "set_dinner_table"
            bimanual = True
            objects = tuple(item for item in objects if item != "table") or ("plate", "cup", "spoon", "fork")
            steps = tuple(TaskStep("place", (item,)) for item in objects)
        elif "open" in text and "drawer" in text:
            goal = "retrieve_from_drawer"
            retrieved = tuple(item for item in objects if item not in {"drawer", "table"}) or ("spoon",)
            steps = (TaskStep("open_drawer", ("drawer",)), TaskStep("retrieve", retrieved, quantity=self._quantity(text)))
        elif "next to" in text:
            if len(objects) < 2:
                raise ValueError("placing next to another object requires two recognized objects")
            goal = "place_next_to"
            constraints.append("preserve_relative_adjacency")
            steps = (TaskStep("place_next_to", objects[:2], relation="next_to"),)
        elif "bring" in text or "take" in text or "get" in text:
            target = tuple(item for item in objects if item not in {"drawer", "table"})
            if not target:
                raise ValueError("retrieval command requires a recognized object")
            goal = "retrieve_object"
            steps = (TaskStep("retrieve", target, quantity=self._quantity(text)),)
        elif "put" in text or "place" in text:
            target = tuple(item for item in objects if item != "table")
            if not target:
                raise ValueError("placement command requires a recognized object")
            goal = "place_object"
            steps = (TaskStep("place", target, relation="on_table" if "table" in text else None),)
        elif "organize" in text or "arrange" in text:
            goal = "organize_table"
            bimanual = True if bimanual or "organize" in text else bimanual
            objects = tuple(item for item in objects if item != "table") or ("plate", "cup", "spoon", "fork")
            steps = (TaskStep("organize", objects),)
        else:
            raise ValueError(f"unsupported dinner-table instruction: {instruction}")
        return TaskCommand(goal, objects, tuple(constraints), bimanual, steps, instruction)

    def _parse_transformers(self, instruction: str) -> TaskCommand:
        """Ask a local Transformers text-generation pipeline for schema JSON."""
        if not self.config.model_name:
            raise RuntimeError("language.model_name is required for Transformers backend")
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("text-generation", model=self.config.model_name, local_files_only=self.config.local_files_only)
        prompt = "Return JSON only with keys goal, objects, constraints, requires_bimanual, steps. Each step has action, objects, relation, quantity, arm. Instruction: " + instruction
        result = self._pipeline(prompt, max_new_tokens=self.config.max_new_tokens, do_sample=False)[0]["generated_text"]
        payload = json.loads(result[result.find("{"):result.rfind("}") + 1])
        steps = tuple(TaskStep(**step) for step in payload.get("steps", []))
        return TaskCommand(str(payload["goal"]), tuple(payload.get("objects", [])), tuple(payload.get("constraints", [])), bool(payload.get("requires_bimanual", False)), steps, instruction, 0.7)

    @staticmethod
    def _objects_in(text: str) -> tuple[str, ...]:
        """Extract unique known object classes without assigning coordinates."""
        found: list[str] = []
        for token in re.findall(r"[a-z]+", text):
            normalized = OBJECT_ALIASES.get(token, token)
            if normalized in OBJECTS and normalized not in found:
                found.append(normalized)
        return tuple(found)

    @staticmethod
    def _quantity(text: str) -> int:
        """Extract a small natural-language quantity."""
        match = re.search(r"\b(two|three|four|\d+)\b", text)
        if not match:
            return 1
        return {"two": 2, "three": 3, "four": 4}.get(match.group(1), int(match.group(1)) if match.group(1).isdigit() else 1)


def parse_instruction(instruction: str, config: dict[str, Any] | None = None) -> TaskCommand:
    """Backward-compatible functional parser entry point."""
    return InstructionParser(config).parse(instruction)