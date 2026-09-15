"""Controller specialization for simulated SO-101 arm A."""

from __future__ import annotations

from typing import Any

from .arm_controller import ArmController


class ArmAController(ArmController):
    """Left-side simulated SO-101 controller."""

    def __init__(self, environment_or_model: Any, data: Any = None, config: dict[str, Any] | None = None) -> None:
        super().__init__("arm_a", environment_or_model, data, config)
