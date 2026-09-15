"""Deterministic official evaluation seed handling."""

from __future__ import annotations

from typing import Any


DEFAULT_SEEDS = tuple(range(10))


def evaluation_seeds(config: dict[str, Any] | None = None) -> list[int]:
    """Return exactly the configured seeds or the official 0-9 default set."""
    values = (config or {}).get("evaluation", config or {}).get("seeds", DEFAULT_SEEDS)
    seeds = [int(seed) for seed in values]
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be a non-empty list of non-negative integers")
    return seeds