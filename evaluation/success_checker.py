"""Task success and placement checks for evaluation episodes."""

from __future__ import annotations

from typing import Any, Sequence


def check_success(distances: Sequence[float], threshold: float = 0.05) -> bool:
    """Check that all target placement distances are within tolerance."""
    if threshold <= 0 or any(distance < 0 for distance in distances):
        raise ValueError("distances must be non-negative and threshold must be positive")
    return bool(distances) and all(distance <= threshold for distance in distances)


def scene_task_success(environment: Any) -> bool:
    """Use the MuJoCo environment's authoritative completion predicate."""
    return bool(environment.is_task_complete())