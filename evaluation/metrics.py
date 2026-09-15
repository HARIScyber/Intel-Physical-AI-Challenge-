"""Episode and aggregate evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EpisodeMetrics:
    """Metrics required for one reproducible evaluation episode."""

    seed: int
    task_success: bool
    episode_length: int
    episode_time: float
    collision_count: int
    grasp_success_rate: float
    placement_success_rate: float
    planning_failures: int
    recovery_count: int
    inference_latency: float
    total_reward: float
    evaluation_mode: str = "medium"
    failure_reason: str | None = None
    left_arm_actions: int = 0
    right_arm_actions: int = 0
    bimanual_actions: int = 0
    single_arm_actions: int = 0
    arm_utilization: float = 0.0
    coordination_success: bool = False
    objects_grasped: int = 0
    objects_placed: int = 0
    objects_verified: int = 0
    failed_grasps: int = 0
    recovery_attempts: int = 0
    safety_violations: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable metric record."""
        return asdict(self)


def success_rate(records: list[EpisodeMetrics]) -> float:
    """Compute task success rate."""
    return _mean([float(record.task_success) for record in records])


def aggregate_metrics(records: list[EpisodeMetrics]) -> dict[str, float]:
    """Compute official aggregate metrics from episode records."""
    if not records:
        raise ValueError("records cannot be empty")
    return {
        "success_rate": success_rate(records),
        "average_time": _mean([record.episode_time for record in records]),
        "average_latency": _mean([record.inference_latency for record in records]),
        "collision_rate": _mean([float(record.collision_count > 0) for record in records]),
        "average_reward": _mean([record.total_reward for record in records]),
    }


def _mean(values: list[float]) -> float:
    """Compute a non-empty arithmetic mean."""
    return sum(values) / len(values) if values else 0.0