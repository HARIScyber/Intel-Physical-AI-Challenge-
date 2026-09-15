"""Official-style ten-seed closed-loop evaluation runner."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control import BimanualController
from language import parse_instruction, reason_about_task
from perception import CameraProcessor
from physical_ai.config import load_config
from planning import TaskPlanner
from policy import ScriptedPolicy
from simulation import BimanualMujocoEnv

try:
    from .metrics import EpisodeMetrics, aggregate_metrics
    from .random_seeds import evaluation_seeds
    from .success_checker import scene_task_success
except ImportError:  # direct ``python evaluation/evaluate.py`` execution
    from evaluation.metrics import EpisodeMetrics, aggregate_metrics
    from evaluation.random_seeds import evaluation_seeds
    from evaluation.success_checker import scene_task_success

LOGGER = logging.getLogger(__name__)


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """Run the configured evaluation seeds and write JSON/CSV reports."""
    if not config:
        raise ValueError("evaluation configuration is required")
    evaluation = config.get("evaluation", config)
    mode = str(evaluation.get("mode", "medium"))
    if mode not in {"easy", "medium", "hard"}:
        raise ValueError("evaluation mode must be easy, medium, or hard")
    seeds = evaluation_seeds(config)
    output_dir = ROOT / evaluation.get("output_dir", "outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[EpisodeMetrics] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        record, failure = _run_episode(config, seed, mode)
        records.append(record)
        if failure is not None:
            failures.append(failure)
        print(f"{seed} | {record.task_success} | {record.episode_time:.3f} | {record.collision_count} | {record.grasp_success_rate:.3f} | {record.inference_latency:.6f}")
    aggregate = aggregate_metrics(records)
    report = {"episodes": [record.as_dict() for record in records], "aggregate": aggregate, "failed_episodes": failures, "seeds": seeds, "evaluation_mode": mode}
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(output_dir / "results.csv", records)
    print(f"Success Rate: {aggregate['success_rate']:.3f}")
    print(f"Average Time: {aggregate['average_time']:.3f}")
    print(f"Average Latency: {aggregate['average_latency']:.6f}")
    print(f"Collision Rate: {aggregate['collision_rate']:.3f}")
    print(f"Average Reward: {aggregate['average_reward']:.3f}")
    return report


def _run_episode(config: Any, seed: int, mode: str) -> tuple[EpisodeMetrics, dict[str, Any] | None]:
    """Run one seed with closed-loop observation after every action chunk."""
    environment_config = config
    if isinstance(config, dict):
        environment_config = SimpleNamespace(simulation=config["simulation"], evaluation={"evaluation": config.get("evaluation", {})})
    environment = BimanualMujocoEnv(environment_config)
    started = time.perf_counter()
    planning_failures = 0
    recovery_count = 0
    collisions = 0
    grasp_attempts = 0
    grasp_successes = 0
    placement_attempts = 0
    placement_successes = 0
    reward = 0.0
    latency_total = 0.0
    failure: dict[str, Any] | None = None
    try:
        observation, reset_info = environment.reset(seed=seed, options={"evaluation_mode": mode})
        perception_config = environment_config.simulation.get("perception", {})
        processor = CameraProcessor(perception_config, ROOT / "outputs" / "evaluation" / "perception")
        scene = processor.perceive(observation, environment.model, environment.data)
        task = reason_about_task(parse_instruction("Set the dinner table."))
        sequence = TaskPlanner({"max_retries": 2}).plan(task, scene)
        policy = ScriptedPolicy()
        policy.reset(sequence)
        controller = BimanualController(environment)
        max_steps = int(environment_config.evaluation.get("evaluation", {}).get("max_steps", 500))
        step_count = 0
        for action_spec in sequence.actions:
            inference_started = time.perf_counter()
            action = policy.predict(scene)
            latency_total += time.perf_counter() - inference_started
            try:
                controller.execute_action(action, duration=1.0 / 20.0)
            except (RuntimeError, ValueError) as exc:
                planning_failures += 1
                recovery_count += 1
                collisions += 1
                failure = {"seed": seed, "mode": mode, "action": action_spec.name, "reason": str(exc), "reset_info": reset_info}
                continue
            step_count += 1
            observation = environment.get_observation()
            scene = processor.perceive(observation, environment.model, environment.data)
            collision = environment.check_collision()
            collisions += int(collision)
            reward += 1.0 if not collision else -1.0
            if action_spec.name in {"pick", "grasp", "retry_grasp"}:
                grasp_attempts += 1
                grasp_successes += int(not collision)
            if action_spec.name == "place":
                placement_attempts += 1
                placement_successes += int(not collision)
            if step_count >= max_steps:
                break
        success = scene_task_success(environment)
        if not success and failure is None:
            failure = {"seed": seed, "mode": mode, "reason": "task_not_complete", "reset_info": reset_info}
        episode_time = time.perf_counter() - started
        record = EpisodeMetrics(seed, success, step_count, episode_time, collisions, _rate(grasp_successes, grasp_attempts), _rate(placement_successes, placement_attempts), planning_failures, recovery_count, _rate(latency_total, max(step_count, 1)), reward, mode, failure["reason"] if failure else None)
        return record, failure
    except Exception as exc:
        failure = {"seed": seed, "mode": mode, "reason": str(exc)}
        episode_time = time.perf_counter() - started
        return EpisodeMetrics(seed, False, 0, episode_time, collisions, 0.0, 0.0, planning_failures + 1, recovery_count + 1, 0.0, reward, mode, str(exc)), failure
    finally:
        environment.close()


def _write_csv(path: Path, records: list[EpisodeMetrics]) -> None:
    """Write one row per evaluation episode."""
    fields = list(records[0].as_dict()) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(record.as_dict() for record in records)


def _rate(successes: int, attempts: int) -> float:
    """Return zero for an action category that was not attempted."""
    return successes / attempts if attempts else 0.0


def main() -> int:
    """Run official ten-seed evaluation from YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("easy", "medium", "hard"), default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    if args.mode is not None:
        config.evaluation["evaluation"]["mode"] = args.mode
    evaluate({"evaluation": config.evaluation["evaluation"], "simulation": config.simulation})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())