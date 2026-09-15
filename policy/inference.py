"""Policy construction and inference helpers with scripted fallback."""

from __future__ import annotations

from typing import Any

from .base_policy import BasePolicy, ScriptedPolicy
from .vla_policy import LeRobotVLA, VLAUnavailableError


class DummyPolicy(BasePolicy):
    """Dependency-free neutral policy for pipeline and integration tests."""

    def load_checkpoint(self, checkpoint: str | None = None) -> None:
        """Ignore checkpoints because the dummy policy has no parameters."""
        del checkpoint

    def reset(self) -> None:
        """Reset the stateless policy."""

    def predict(self, observation: Any, instruction: str | None = None) -> dict[str, Any]:
        """Return neutral bounded commands for both arms."""
        if observation is None or instruction is None:
            raise ValueError("observation and instruction are required")
        import numpy as np
        return {"arm_a": np.zeros(6), "arm_b": np.zeros(6), "gripper_a": 0.0, "gripper_b": 0.0}


def create_policy(config: dict[str, Any] | None = None) -> BasePolicy:
    """Create the configured policy without requiring a checkpoint."""
    settings = config or {}
    policy_type = str(settings.get("type", "scripted")).lower()
    if policy_type in {"smolvla", "lerobot", "vla"}:
        if not settings.get("checkpoint") and bool(settings.get("fallback_to_scripted", True)):
            return ScriptedPolicy(settings)
        policy: BasePolicy = LeRobotVLA(settings)
        if settings.get("checkpoint") and settings.get("pretrained", True):
            try:
                policy.load_checkpoint(str(settings["checkpoint"]))
            except VLAUnavailableError:
                if bool(settings.get("fallback_to_scripted", True)):
                    return ScriptedPolicy(settings)
                raise
        return policy
    if policy_type == "dummy":
        return DummyPolicy(settings)
    return ScriptedPolicy(settings)


def infer(policy: BasePolicy, observation: Any, instruction: str | None = None, config: dict | None = None) -> Any:
    """Run one model-independent policy inference."""
    del config
    if policy is None:
        raise ValueError("policy is required")
    return policy.predict(observation, instruction)


class ClosedLoopPipeline:
    """Observe, evaluate, act, and record one action chunk at a time."""

    def __init__(self, environment: Any, processor: Any, planner: Any, policy: BasePolicy, controller: Any, run_dir: Any, instruction: str) -> None:
        self.environment = environment
        self.processor = processor
        self.planner = planner
        self.policy = policy
        self.controller = controller
        self.run_dir = run_dir
        self.instruction = instruction
        self.trajectory: list[dict[str, Any]] = []

    def run(self, seed: int, max_chunks: int = 40) -> dict[str, Any]:
        """Run closed-loop inference until verification succeeds or the budget ends."""
        import json
        import time
        transition_log = (self.run_dir / "logs" / "transitions.jsonl").open("a", encoding="utf-8")
        observation, reset_info = self.environment.reset(seed=seed)
        scene = self.processor.perceive(observation, self.environment.model, self.environment.data)
        from language import parse_instruction, reason_about_task
        task = reason_about_task(parse_instruction(self.instruction))
        sequence = self.planner.plan(task, scene)
        if hasattr(self.policy, "reset"):
            try:
                self.policy.reset(sequence)
            except TypeError:
                self.policy.reset()
        started = time.perf_counter()
        success = False
        collision_count = 0
        for chunk_index in range(max_chunks):
            before = scene.as_dict()
            action = self.policy.predict(observation, self.instruction)
            if not isinstance(action, dict):
                raise ValueError("policy must return a dual-arm action mapping")
            action = {key: value for key, value in action.items() if not key.startswith("_")}
            try:
                controller_state = self.controller.execute_action(action, duration=1.0 / 20.0)
                observation = self.environment.get_observation()
                collision = bool(self.environment.check_collision())
            except (RuntimeError, ValueError) as exc:
                collision = True
                collision_count += 1
                controller_state = {"error": str(exc)}
                observation = self.environment.get_observation()
            scene = self.processor.perceive(observation, self.environment.model, self.environment.data)
            after = scene.as_dict()
            success = bool(after.get("robot_state", {}).get("task_state", {}).get("complete", False))
            record = {"chunk_index": chunk_index, "action": _jsonable(action), "state_before": _jsonable(before), "controller_state": _jsonable(controller_state), "state_after": _jsonable(after), "collision": collision, "timestamp": time.time()}
            self.trajectory.append(record)
            transition_log.write(json.dumps(record) + "\n")
            transition_log.flush()
            for camera_name, frame in observation.get("camera_images", {}).items():
                import cv2
                cv2.imwrite(str(self.run_dir / "frames" / f"chunk_{chunk_index:04d}_{camera_name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if success:
                break
        transition_log.close()
        metrics = {"success": success, "chunks": len(self.trajectory), "collisions": collision_count, "duration_seconds": time.perf_counter() - started, "seed": seed, "instruction": self.instruction, "reset_info": reset_info, "policy": type(self.policy).__name__}
        (self.run_dir / "trajectory.json").write_text(json.dumps(_jsonable(self.trajectory), indent=2), encoding="utf-8")
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


def _jsonable(value: Any) -> Any:
    """Convert NumPy and nested dataclass-like values for run logs."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value