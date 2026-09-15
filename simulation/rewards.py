"""Reward terms for future task-specific learning experiments."""

import logging

LOGGER = logging.getLogger(__name__)


def placement_reward(distance: float, threshold: float = 0.05) -> float:
    """Return a bounded shaping reward based on placement distance."""
    if distance < 0 or threshold <= 0:
        raise ValueError("distance must be non-negative and threshold must be positive")
    return float(max(0.0, 1.0 - distance / threshold))
