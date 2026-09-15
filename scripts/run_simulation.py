"""Run the headless MuJoCo dinner-table simulation."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physical_ai.config import load_config
from simulation import BimanualMujocoEnv

LOGGER = logging.getLogger(__name__)

def main() -> int:
    """Load configuration, reset the scene, and advance physics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs", help="Configuration directory")
    parser.add_argument("--duration", type=float, default=None, help="Override duration in seconds")
    parser.add_argument("--headless", action="store_true", help="Disable the interactive MuJoCo viewer")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config_dir, ROOT)
        env = BimanualMujocoEnv(config)
        _, info = env.reset(seed=int(config.simulation["simulation"].get("seed", 0)))
        show_viewer = bool(config.simulation["simulation"].get("render", True)) and not args.headless
        if show_viewer:
            env.render()
        duration = args.duration or float(config.simulation["simulation"].get("duration_seconds", 5.0))
        steps = max(1, int(duration / float(config.simulation["simulation"].get("timestep", 0.002))))
        for _ in range(steps):
            scripted_action = {
                "arm_a": [0.0, -0.25, 0.45, 0.0, -0.2, 0.0],
                "arm_b": [0.0, -0.25, 0.45, 0.0, -0.2, 0.0],
                "gripper_a": 0.0,
                "gripper_b": 0.0,
            }
            _, _, terminated, truncated, _ = env.step(scripted_action)
            if show_viewer and env._viewer is not None and not env._viewer.is_running():
                break
            if terminated or truncated:
                break
        env.close()
        LOGGER.info("Simulation completed: %d steps, reset=%s", steps, info)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        LOGGER.error("Simulation failed: %s", exc)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
