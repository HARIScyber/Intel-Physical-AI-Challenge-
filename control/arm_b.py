"""Controller specialization for simulated SO-101 arm B."""

from __future__ import annotations

from typing import Any

from .arm_controller import ArmController


class ArmBController(ArmController):
    """Right-side simulated SO-101 controller."""

    def __init__(self, environment_or_model: Any, data: Any = None, config: dict[str, Any] | None = None) -> None:
        super().__init__("arm_b", environment_or_model, data, config)
