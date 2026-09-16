"""Task success and placement checks for evaluation episodes."""

from __future__ import annotations

from typing import Any, Sequence

from evaluation.task_verifier import TaskVerifier, TaskVerificationResult


def check_success(distances: Sequence[float], threshold: float = 0.05) -> bool:
    """Check that all target placement distances are within tolerance."""
    if threshold <= 0 or any(distance < 0 for distance in distances):
        raise ValueError("distances must be non-negative and threshold must be positive")
    return bool(distances) and all(distance <= threshold for distance in distances)


def scene_task_success(environment: Any) -> bool:
    """Use the MuJoCo environment's authoritative completion predicate.

    DEPRECATED: Use verify_task() for complete verification with manipulation history.
    """
    return bool(environment.is_task_complete())


def verify_task_complete(environment: Any, task_command: Any = None, manipulation_history: dict[str, set[str]] | None = None) -> TaskVerificationResult:
    """Authoritative task verification using TaskVerifier."""
    verifier = TaskVerifier(environment, task_command)
    return verifier.verify(manipulation_history)