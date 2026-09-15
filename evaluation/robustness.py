"""Seeded robustness evaluation across domain-randomization difficulty modes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, asdict
from typing import Any

LOGGER = logging.getLogger(__name__)
MODES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class RobustnessResult:
    """One reproducible evaluation outcome."""

    seed: int
    mode: str
    success: bool
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        return asdict(self)


def run_trials(trial: Callable[[int], bool], seeds: list[int]) -> list[bool]:
    """Backward-compatible deterministic trial runner."""
    if trial is None or not seeds:
        raise ValueError("trial and seeds are required")
    return [bool(trial(seed)) for seed in seeds]


def evaluate_modes(trial: Callable[[int, str], bool], seeds: list[int], modes: tuple[str, ...] = MODES) -> list[RobustnessResult]:
    """Run every seed/mode pair and retain enough information to reproduce failures."""
    if trial is None or not seeds:
        raise ValueError("trial and seeds are required")
    invalid = set(modes) - set(MODES)
    if invalid:
        raise ValueError(f"unsupported robustness modes: {sorted(invalid)}")
    results: list[RobustnessResult] = []
    for mode in modes:
        for seed in seeds:
            if seed < 0:
                raise ValueError("seeds must be non-negative")
            try:
                success = bool(trial(seed, mode))
                results.append(RobustnessResult(seed, mode, success, {}))
            except Exception as exc:
                LOGGER.exception("Robustness trial failed: mode=%s seed=%s", mode, seed)
                results.append(RobustnessResult(seed, mode, False, {"error": str(exc)}))
    return results


def replay_seed(trial: Callable[[int, str], bool], seed: int, mode: str) -> RobustnessResult:
    """Replay one recorded failure using its exact seed and mode."""
    return evaluate_modes(trial, [seed], (mode,))[0]