"""Generate local scripted MuJoCo demonstrations with randomized episodes."""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from language import parse_instruction, reason_about_task
from perception import CameraProcessor
from physical_ai.config import load_config
from planning import Action, ActionSequence, TaskPlanner
from policy import ScriptedPolicy
from simulation import BimanualMujocoEnv
from simulation.randomization import language_variant

LOGGER = logging.getLogger(__name__)

def generate(output_dir: str | Path, config: dict[str, Any], episodes: int | None = None, mode: str = "medium") -> Path:
    """Generate deterministic local episodes under ``output_dir``."""
    if not config:
        raise ValueError("configuration is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    count = int(episodes if episodes is not None else config.get("demonstrations", {}).get("episodes", 1))
    if count < 1:
        raise ValueError("episodes must be positive")
    instruction = str(config.get("demonstrations", {}).get("language_instruction", "Set the dinner table."))
    project_config = load_config(project_root=ROOT)
    for episode_index in range(count):
        _generate_episode(destination, project_config, episode_index, instruction, mode)
    return destination

def _generate_episode(destination: Path, config: Any, episode_index: int, instruction: str, mode: str = "medium") -> None:
    """Record one episode as image files, arrays, and metadata."""
    environment = BimanualMujocoEnv(config)
    started = time.time()
    episode_id = f"episode_{episode_index:04d}"
    episode_dir = destination / episode_id
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        seed = int(config.training.get("training", {}).get("seed", 0)) + episode_index
        observation, reset_info = environment.reset(seed=seed, options={"evaluation_mode": mode})
        instruction = language_variant(np.random.default_rng(seed), mode)
        perception_config = config.simulation.get("perception", {})
        processor = CameraProcessor(perception_config, episode_dir / "perception_debug")
        scene = processor.perceive(observation, environment.model, environment.data, perception_config.get("camera", "overhead"))
        command = reason_about_task(parse_instruction(instruction))
        sequence = TaskPlanner({"max_retries": 2}).plan(command, scene)
        sequence = ActionSequence(sequence.actions[:-2] + (Action("coordinate", 0.5, arm="both"), Action("handoff", 1.0, ("spoon",), "both"), Action("verify", 0.5), Action("finish", 0.1)), sequence.goal)
        policy = ScriptedPolicy()
        policy.reset(sequence)
        timestep = float(config.simulation["simulation"].get("timestep", 0.002))
        frequency = float(config.simulation["simulation"].get("control_frequency", 20))
        physics_steps = max(1, int(round(1.0 / frequency / timestep)))
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        timestamps: list[float] = []
        image_paths: list[dict[str, str]] = []
        action_names: list[str] = []
        step_index = 0
        for action in sequence.actions:
            command_action = policy.predict(scene)
            command_action["_physics_steps"] = physics_steps
            repeats = max(1, int(round(action.duration * frequency)))
            for _ in range(repeats):
                frame_names: dict[str, str] = {}
                for camera_name, frame in observation["camera_images"].items():
                    filename = f"frame_{step_index:06d}_{camera_name}.png"
                    path = frames_dir / filename
                    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                        raise OSError(f"could not write demonstration frame: {path}")
                    frame_names[camera_name] = str(Path("frames") / filename)
                states.append(_state_vector(observation))
                actions.append(_action_vector(command_action))
                timestamps.append(float(step_index / frequency))
                image_paths.append(frame_names)
                action_names.append(action.name)
                observation, _, _, truncated, _ = environment.step(command_action)
                step_index += 1
                if truncated:
                    break
            scene = processor.perceive(observation, environment.model, environment.data, perception_config.get("camera", "overhead"))
        state_array = np.asarray(states, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
        np.savez_compressed(episode_dir / "data.npz", observation_state=state_array, action=action_array, timestamp=np.asarray(timestamps, dtype=np.float64))
        metadata = {"episode_id": episode_id, "seed": seed, "language_instruction": instruction, "task": command.as_dict(), "reset_info": reset_info, "step_count": len(states), "episode_duration_seconds": time.time() - started, "action_names": action_names, "image_paths": image_paths, "camera_names": ["overhead", "front"], "state_shape": list(state_array.shape), "action_shape": list(action_array.shape)}
        (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info("Generated %s with %d steps", episode_id, len(states))
    finally:
        environment.close()

def _state_vector(observation: dict[str, Any]) -> np.ndarray:
    """Flatten robot, object, orientation, and drawer state into finite values."""
    values: list[float] = list(np.asarray(observation["robot_joint_positions"], dtype=np.float32).reshape(-1)) + list(np.asarray(observation["robot_joint_velocities"], dtype=np.float32).reshape(-1))
    for name in sorted(observation["object_positions"]):
        values.extend(observation["object_positions"][name])
        values.extend(observation["object_orientations"][name])
    values.append(float(observation["drawer_state"]["position"]))
    return np.asarray(values, dtype=np.float32)

def _action_vector(action: dict[str, Any]) -> np.ndarray:
    """Convert a scripted command to arm joints, grippers, and drawer."""
    return np.asarray(list(action["arm_a"]) + list(action["arm_b"]) + [action["gripper_a"], action["gripper_b"], action.get("drawer", 0.0) or 0.0], dtype=np.float32)

def main() -> int:
    """Generate the configured small local dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=("easy", "medium", "hard"), default="medium")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    project = load_config()
    demonstrations = project.training.get("demonstrations", {})
    output = args.output_dir or demonstrations.get("output_dir", "datasets/raw")
    generate(ROOT / output, project.training, args.episodes, args.mode)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())