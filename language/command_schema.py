"""Typed schemas for natural-language dinner-table tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskStep:
    """One executable semantic step before motion planning."""

    action: str
    objects: tuple[str, ...] = ()
    relation: str | None = None
    quantity: int = 1
    arm: str | None = None


@dataclass(frozen=True)
class TaskCommand:
    """Normalized task representation consumed by planning and policy layers."""

    goal: str
    objects: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    requires_bimanual: bool = False
    steps: tuple[TaskStep, ...] = ()
    raw_instruction: str = ""
    confidence: float = 1.0

    @property
    def intent(self) -> str:
        """Backward-compatible alias for the normalized goal."""
        return self.goal

    @property
    def target(self) -> str:
        """Backward-compatible natural-language target."""
        return self.raw_instruction or self.goal

    @property
    def parameters(self) -> dict[str, str]:
        """Backward-compatible parameter mapping."""
        return {"objects": ",".join(self.objects), "requires_bimanual": str(self.requires_bimanual)}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable task representation."""
        return asdict(self)


@dataclass(frozen=True)
class LanguageConfig:
    """Configuration for deterministic and local Transformers backends."""

    backend: str = "rules"
    model_name: str = ""
    local_files_only: bool = True
    max_new_tokens: int = 256

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "LanguageConfig":
        """Build configuration without requiring a model or network access."""
        values = config or {}
        return cls(str(values.get("backend", "rules")), str(values.get("model_name", "")), bool(values.get("local_files_only", True)), int(values.get("max_new_tokens", 256)))