from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control import BimanualController
from optimization.convert_openvino import (
    INPUT_SHAPES,
    BenchmarkSample,
    SmolVLABenchmarkForward,
    convert_model,
    load_smolvla_model,
    make_benchmark_sample,
)
from perception import CameraProcessor
from physical_ai.config import ProjectConfig, load_config
from simulation import BimanualMujocoEnv

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    backend: str
    model_name: str
    precision: str
    device: str
    status: str
    model_load_time_seconds: float | None
    warmup_time_seconds: float | None
    single_inference_latency_seconds: float | None
    average_latency_seconds: float | None
    p50_latency_seconds: float | None
    p95_latency_seconds: float | None
    throughput_samples_per_second: float | None
    iterations: int
    output_max_abs_diff: float | None = None
    output_mean_abs_diff: float | None = None
    device_precision_hint: str | None = None
    error: str | None = None


@dataclass
class EpisodeMetrics:
    seed: int
    task_success: bool
    episode_duration_seconds: float
    action_latency_seconds: float
    environment_task_complete: bool
    collision: bool
    status: str
    error: str | None = None


@dataclass
class TaskMetrics:
    backend: str
    precision: str
    device: str
    status: str
    task_success: float | None
    task_success_rate: float | None
    environment_task_complete_rate: float | None
    average_episode_duration_seconds: float | None
    average_action_latency_seconds: float | None
    task_episodes: int
    task_steps: int
    task_seed: int
    episodes: list[dict[str, Any]]
    error: str | None = None


class Backend:
    name: str
    precision: str
    device: str
    load_time_seconds: float
    device_precision_hint: str | None = None

    def infer(self, sample: BenchmarkSample) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None


class PyTorchBackend(Backend):
    def __init__(self, checkpoint: str, device: str = "cpu"):
        started = time.perf_counter()
        model, _ = load_smolvla_model(checkpoint, device)
        self.wrapper = SmolVLABenchmarkForward(model).eval()
        self.load_time_seconds = time.perf_counter() - started
        self.name = "PyTorch"
        self.precision = "fp32"
        self.device = device
        self.device_precision_hint = "float32"

    def infer(self, sample: BenchmarkSample) -> np.ndarray:
        with torch.inference_mode():
            output = self.wrapper(*sample.torch_inputs())
        return output.detach().cpu().numpy().copy()


class OpenVINOBackend(Backend):
    def __init__(self, model_path: Path, precision: str, device: str):
        started = time.perf_counter()
        core = ov.Core()
        ov_device = device.upper() if device.lower() in {"cpu", "gpu"} else device
        model = core.read_model(str(model_path))
        model = self._reshape_static(model)
        self.compiled_model = core.compile_model(model, ov_device)
        self.request = self.compiled_model.create_infer_request()
        self.load_time_seconds = time.perf_counter() - started
        self.name = "OpenVINO"
        self.precision = precision
        self.device = ov_device
        try:
            self.device_precision_hint = str(core.get_property(ov_device, "INFERENCE_PRECISION_HINT")).strip("<> '")
        except RuntimeError:
            self.device_precision_hint = None

    @staticmethod
    def _reshape_static(model: ov.Model) -> ov.Model:
        from optimization.convert_openvino import INPUT_SHAPES

        model.reshape({port: ov.PartialShape(shape) for port, shape in zip(model.inputs, INPUT_SHAPES)})
        return model

    def infer(self, sample: BenchmarkSample) -> np.ndarray:
        output = self.request.infer(sample.openvino_inputs())
        values = list(output.values())
        if not values:
            raise RuntimeError("OpenVINO model returned no outputs")
        return np.asarray(values[0]).copy()


def _sample_digest(sample: BenchmarkSample) -> str:
    digest = hashlib.sha256()
    values = (
        ("image", sample.image),
        ("img_mask", sample.img_mask),
        ("lang_tokens", sample.lang_tokens),
        ("lang_masks", sample.lang_masks),
        ("state", sample.state),
        ("x_t", sample.x_t),
        ("timestep", sample.timestep),
    )
    for name, value in values:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _output_diff(reference: np.ndarray, actual: np.ndarray) -> tuple[float | None, float | None]:
    if reference.shape != actual.shape:
        return None, None
    difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return float(difference.max(initial=0.0)), float(difference.mean())


def _benchmark_inference(
    backend: Backend,
    sample: BenchmarkSample,
    warmup_iterations: int,
    measured_iterations: int,
    reference_output: np.ndarray | None = None,
) -> tuple[InferenceMetrics, np.ndarray]:
    try:
        reference = backend.infer(sample)
        started = time.perf_counter()
        for _ in range(warmup_iterations):
            backend.infer(sample)
        warmup_time = time.perf_counter() - started

        started = time.perf_counter()
        single_output = backend.infer(sample)
        single_latency = time.perf_counter() - started

        latencies: list[float] = []
        for _ in range(measured_iterations):
            started = time.perf_counter()
            backend.infer(sample)
            latencies.append(time.perf_counter() - started)

        average = float(np.mean(latencies))
        max_diff, mean_diff = _output_diff(reference, single_output)
        if reference_output is not None:
            max_diff, mean_diff = _output_diff(reference_output, single_output)
        metrics = InferenceMetrics(
            backend=backend.name,
            model_name=backend.name,
            precision=backend.precision,
            device=backend.device,
            status="ready",
            model_load_time_seconds=float(backend.load_time_seconds),
            warmup_time_seconds=float(warmup_time),
            single_inference_latency_seconds=float(single_latency),
            average_latency_seconds=average,
            p50_latency_seconds=float(np.percentile(latencies, 50)),
            p95_latency_seconds=float(np.percentile(latencies, 95)),
            throughput_samples_per_second=float(measured_iterations / sum(latencies)),
            iterations=measured_iterations,
            output_max_abs_diff=max_diff,
            output_mean_abs_diff=mean_diff,
            device_precision_hint=backend.device_precision_hint,
        )
        return metrics, single_output
    except Exception as exc:
        return (
            InferenceMetrics(
                backend=backend.name,
                model_name=backend.name,
                precision=backend.precision,
                device=backend.device,
                status="failed",
                model_load_time_seconds=float(backend.load_time_seconds),
                warmup_time_seconds=None,
                single_inference_latency_seconds=None,
                average_latency_seconds=None,
                p50_latency_seconds=None,
                p95_latency_seconds=None,
                throughput_samples_per_second=None,
                iterations=measured_iterations,
                device_precision_hint=backend.device_precision_hint,
                error=str(exc),
            ),
            np.asarray([], dtype=np.float32),
        )


def _action_from_output(output: np.ndarray, observation: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    if values.size < 12 or not np.isfinite(values[:12]).all():
        raise ValueError("model output does not contain a finite 12-value action")
    current = np.asarray(observation.get("robot_joint_positions", ()), dtype=np.float64).reshape(-1)
    if current.size < 12:
        raise ValueError("environment observation does not contain 12 joint positions")
    delta = np.tanh(values[:12]) * 0.01
    return {
        "arm_a": current[:6] + delta[:6],
        "arm_b": current[6:12] + delta[6:12],
        "gripper_a": 0.5,
        "gripper_b": 0.5,
    }


def _run_task_benchmark(
    backend: Backend,
    sample: BenchmarkSample,
    config: ProjectConfig,
    seed: int,
    episodes: int,
    steps: int,
) -> TaskMetrics:
    episode_records: list[dict[str, Any]] = []
    try:
        from language import parse_instruction, reason_about_task
        from planning import TaskPlanner

        for episode_index in range(episodes):
            environment = BimanualMujocoEnv(config)
            controller: BimanualController | None = None
            episode_start = time.perf_counter()
            success = False
            collision = False
            action_latencies: list[float] = []
            environment_complete = False
            error: str | None = None
            try:
                observation, _ = environment.reset(seed=seed + episode_index)
                processor = CameraProcessor(
                    config.simulation.get("perception", {}),
                    ROOT / "outputs" / "benchmarks" / "perception",
                )
                scene = processor.perceive(
                    observation,
                    environment.model,
                    environment.data,
                    config.simulation.get("camera", "overhead"),
                )
                task = reason_about_task(parse_instruction("Set the dinner table."))
                TaskPlanner({"max_retries": 1}).plan(task, scene)
                controller = BimanualController(environment)
                for _ in range(steps):
                    started = time.perf_counter()
                    output = backend.infer(sample)
                    action_latencies.append(time.perf_counter() - started)
                    action = _action_from_output(output, observation)
                    controller.execute_action(action, duration=0.05)
                    observation = environment.get_observation()
                    collision = collision or environment.check_collision()
                environment_complete = bool(environment.is_task_complete())
                success = bool(action_latencies) and not collision and environment_complete and all(
                    np.isfinite(value) for value in action_latencies
                )
            except Exception as exc:
                error = str(exc)
                success = False
            finally:
                duration = time.perf_counter() - episode_start
                if controller is not None:
                    del controller
                environment.close()
            episode_records.append(
                {
                    "seed": seed + episode_index,
                    "task_success": bool(success),
                    "episode_duration_seconds": float(duration),
                    "action_latency_seconds": float(np.mean(action_latencies)) if action_latencies else None,
                    "environment_task_complete": bool(environment_complete),
                    "collision": bool(collision),
                    "status": "completed" if error is None else "failed",
                    "error": error,
                }
            )

        successes = [record["task_success"] for record in episode_records]
        durations = [record["episode_duration_seconds"] for record in episode_records]
        latencies = [record["action_latency_seconds"] for record in episode_records if record["action_latency_seconds"] is not None]
        environment_complete_rate = float(np.mean([record["environment_task_complete"] for record in episode_records])) if episode_records else None
        return TaskMetrics(
            backend=backend.name,
            precision=backend.precision,
            device=backend.device,
            status="ready",
            task_success=float(np.mean(successes)) if successes else None,
            task_success_rate=float(np.mean(successes)) if successes else None,
            environment_task_complete_rate=environment_complete_rate,
            average_episode_duration_seconds=float(np.mean(durations)) if durations else None,
            average_action_latency_seconds=float(np.mean(latencies)) if latencies else None,
            task_episodes=episodes,
            task_steps=steps,
            task_seed=seed,
            episodes=episode_records,
        )
    except Exception as exc:
        return TaskMetrics(
            backend=backend.name,
            precision=backend.precision,
            device=backend.device,
            status="failed",
            task_success=None,
            task_success_rate=None,
            average_episode_duration_seconds=None,
            average_action_latency_seconds=None,
            task_episodes=episodes,
            task_steps=steps,
            task_seed=seed,
            episodes=episode_records,
            error=str(exc),
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "backend",
        "model_name",
        "precision",
        "device",
        "status",
        "model_load_time_seconds",
        "warmup_time_seconds",
        "single_inference_latency_seconds",
        "average_latency_seconds",
        "p50_latency_seconds",
        "p95_latency_seconds",
        "throughput_samples_per_second",
        "iterations",
        "output_max_abs_diff",
        "output_mean_abs_diff",
        "device_precision_hint",
        "task_success_rate",
        "environment_task_complete_rate",
        "average_episode_duration_seconds",
        "average_action_latency_seconds",
        "task_episodes",
        "task_steps",
        "task_seed",
        "task_status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_seconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    rows = report["results"]
    lines = [
        "# OpenVINO Benchmark Report",
        "",
        f"Generated: `{metadata['generated_at']}`",
        "",
        "## Methodology",
        "",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Device requested: `{metadata['device']}`",
        f"- Identical sample seed: `{metadata['sample_seed']}`",
        f"- Sample SHA-256: `{metadata['sample_sha256']}`",
        f"- Warmup iterations: `{metadata['warmup_iterations']}`",
        f"- Measured inference iterations: `{metadata['measured_iterations']}`",
        f"- Task episodes: `{metadata['task_episodes']}`",
        f"- Task steps per episode: `{metadata['task_steps']}`",
        "",
        "The benchmark uses one fixed image, mask, token, state, action-noise, and timestep sample for every backend. The exported graph is the same SmolVLA forward computation with KV-cache reuse disabled so the PyTorch and OpenVINO paths receive identical tensors.",
        "",
        "The task benchmark executes one bounded dual-arm action in the MuJoCo environment per episode. Task success means a finite model action was executed without a collision and the simulator's full dinner-table predicate is satisfied; `environment_task_complete` records that predicate separately.",
        "",
        "## Inference Results",
        "",
        "| Backend | Precision | Device | Load (s) | Warmup (s) | Single (s) | Avg (s) | p50 (s) | p95 (s) | Throughput (samples/s) | Status |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {precision} | {device} | {load} | {warmup} | {single} | {avg} | {p50} | {p95} | {throughput} | {status} |".format(
                backend=row["backend"],
                precision=row["precision"],
                device=row["device"],
                load=_format_seconds(row.get("model_load_time_seconds")),
                warmup=_format_seconds(row.get("warmup_time_seconds")),
                single=_format_seconds(row.get("single_inference_latency_seconds")),
                avg=_format_seconds(row.get("average_latency_seconds")),
                p50=_format_seconds(row.get("p50_latency_seconds")),
                p95=_format_seconds(row.get("p95_latency_seconds")),
                throughput=_format_rate(row.get("throughput_samples_per_second")),
                status=row.get("status", "unknown"),
            )
        )

    lines.extend(
        [
            "",
            "## End-to-End Task Results",
            "",
            "| Backend | Precision | Task Success Rate | Environment Complete | Avg Episode Duration (s) | Avg Action Latency (s) | Episodes | Status |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {backend} | {precision} | {success} | {env_complete} | {duration} | {latency} | {episodes} | {status} |".format(
                backend=row["backend"],
                precision=row["precision"],
                success=_format_rate(row.get("task_success_rate")),
                env_complete=_format_rate(row.get("environment_task_complete_rate")),
                duration=_format_seconds(row.get("average_episode_duration_seconds")),
                latency=_format_seconds(row.get("average_action_latency_seconds")),
                episodes=row.get("task_episodes", "N/A"),
                status=row.get("task_status", row.get("status", "unknown")),
            )
        )

    lines.extend(["", "## Precision Artifacts", ""])
    for artifact in report["openvino_artifacts"]:
        lines.append(
            f"- **{artifact['precision'].upper()}**: `{artifact['status']}` — {artifact['message']} (`{artifact.get('size_bytes')}` bytes)."
        )

    reference = next((row for row in rows if row["backend"] == "PyTorch"), None)
    if reference and reference.get("average_latency_seconds"):
        lines.extend(["", "## Analysis", ""])
        for row in rows:
            if row["backend"] == "PyTorch" or not row.get("average_latency_seconds"):
                continue
            ratio = row["average_latency_seconds"] / reference["average_latency_seconds"]
            lines.append(
                f"- OpenVINO {row['precision'].upper()} average latency is {ratio:.2f}x the PyTorch baseline on `{row['device']}`."
            )
        fp16 = next((row for row in rows if row["backend"] == "OpenVINO" and row["precision"] == "fp16"), None)
        if fp16:
            lines.append(
                f"- CPU precision hint for FP16: `{fp16.get('device_precision_hint')}`; FP16 may be emulated when the CPU reports only FP32 inference precision."
            )

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "python optimization/intel_benchmark.py lerobot/smolvla_base --iterations 3 --warmup 1 --task-episodes 1 --task-steps 1",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_full_benchmark(
    checkpoint: str,
    output_dir: str | Path = "outputs/benchmarks",
    device: str = "cpu",
    warmup_iterations: int = 1,
    measured_iterations: int = 3,
    task_episodes: int = 1,
    task_steps: int = 1,
    task_seed: int = 7,
    sample_seed: int = 123,
    force_convert: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config()
    sample = make_benchmark_sample(sample_seed)
    artifacts = convert_model(checkpoint, ROOT / "outputs" / "openvino", device, force_convert)
    supported_openvino = [artifact for artifact in artifacts if artifact["supported"] and artifact["status"] == "ready"]

    results: list[dict[str, Any]] = []
    pytorch_backend = PyTorchBackend(checkpoint, device)
    try:
        pytorch_inference, reference_output = _benchmark_inference(
            pytorch_backend,
            sample,
            warmup_iterations,
            measured_iterations,
        )
        pytorch_task = _run_task_benchmark(
            pytorch_backend,
            sample,
            config,
            task_seed,
            task_episodes,
            task_steps,
        )
        pytorch_row = asdict(pytorch_inference)
        pytorch_row.update(
            {
                "task_success_rate": pytorch_task.task_success_rate,
                "average_episode_duration_seconds": pytorch_task.average_episode_duration_seconds,
                "average_action_latency_seconds": pytorch_task.average_action_latency_seconds,
                "task_episodes": pytorch_task.task_episodes,
                "task_steps": pytorch_task.task_steps,
                "task_seed": pytorch_task.task_seed,
                "task_status": pytorch_task.status,
            }
        )
        results.append(pytorch_row)
    finally:
        pytorch_backend.close()

    for artifact in supported_openvino:
        backend = OpenVINOBackend(ROOT / artifact["path"], artifact["precision"], device)
        try:
            inference, _ = _benchmark_inference(
                backend,
                sample,
                warmup_iterations,
                measured_iterations,
                reference_output,
            )
            task = _run_task_benchmark(
                backend,
                sample,
                config,
                task_seed,
                task_episodes,
                task_steps,
            )
                row = asdict(inference)
                row.update(
                    {
                        "model_name": artifact["path"],
                        "task_success_rate": task.task_success_rate,
                        "environment_task_complete_rate": task.environment_task_complete_rate,
                        "average_episode_duration_seconds": task.average_episode_duration_seconds,
                        "average_action_latency_seconds": task.average_action_latency_seconds,
                        "task_episodes": task.task_episodes,
                        "task_steps": task.task_steps,
                        "task_seed": task.task_seed,
                        "task_status": task.status,
                    }
                )
            results.append(row)
        finally:
            backend.close()

    for artifact in artifacts:
        if artifact["supported"] and artifact["status"] == "ready":
            continue
        results.append(
            {
                "backend": "OpenVINO",
                "model_name": artifact["path"],
                "precision": artifact["precision"],
                "device": device.upper(),
                "status": artifact["status"],
                "model_load_time_seconds": None,
                "warmup_time_seconds": None,
                "single_inference_latency_seconds": None,
                "average_latency_seconds": None,
                "p50_latency_seconds": None,
                "p95_latency_seconds": None,
                "throughput_samples_per_second": None,
                "iterations": measured_iterations,
                "output_max_abs_diff": None,
                "output_mean_abs_diff": None,
                "device_precision_hint": None,
                "task_success_rate": None,
                "average_episode_duration_seconds": None,
                "average_action_latency_seconds": None,
                "task_episodes": task_episodes,
                "task_steps": task_steps,
                "task_seed": task_seed,
                "task_status": artifact["status"],
                "error": artifact["message"],
            }
        )

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint,
            "device": device,
            "sample_seed": sample_seed,
            "sample_sha256": _sample_digest(sample),
            "sample_shapes": {
                "image": list(INPUT_SHAPES[0]),
                "img_mask": list(INPUT_SHAPES[1]),
                "lang_tokens": list(INPUT_SHAPES[2]),
                "lang_masks": list(INPUT_SHAPES[3]),
                "state": list(INPUT_SHAPES[4]),
                "x_t": list(INPUT_SHAPES[5]),
                "timestep": list(INPUT_SHAPES[6]),
            },
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "task_episodes": task_episodes,
            "task_steps": task_steps,
            "task_seed": task_seed,
        },
        "openvino_artifacts": artifacts,
        "results": results,
    }

    (output_path / "results.json").write_text(
        json.dumps(report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_csv(output_path / "results.csv", results)
    _write_markdown(ROOT / "docs" / "openvino.md", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and OpenVINO SmolVLA inference and task execution")
    parser.add_argument("checkpoint", nargs="?", default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--task-episodes", type=int, default=1)
    parser.add_argument("--task-steps", type=int, default=1)
    parser.add_argument("--task-seed", type=int, default=7)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--force-convert", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run_full_benchmark(
        args.checkpoint,
        args.output_dir,
        args.device,
        args.warmup,
        args.iterations,
        args.task_episodes,
        args.task_steps,
        args.task_seed,
        args.sample_seed,
        args.force_convert,
    )
    print(json.dumps(report["results"], indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
