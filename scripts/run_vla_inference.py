"""Run closed-loop language-to-MuJoCo inference and record a timestamped run."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control import BimanualController
from perception import CameraProcessor
from physical_ai.config import load_config
from planning import TaskPlanner
from policy import ClosedLoopPipeline, create_policy
from simulation import BimanualMujocoEnv


def main() -> int:
    """Execute one closed-loop VLA or scripted-fallback run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="Set the dinner table.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=40)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "outputs" / "runs" / timestamp
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    environment = BimanualMujocoEnv(config)
    try:
        perception_config = config.simulation.get("perception", {})
        processor = CameraProcessor(perception_config, run_dir / "frames")
        policy_config = config.training.get("policy", {})
        policy = create_policy(policy_config)
        planner = TaskPlanner({"max_retries": 2})
        controller = BimanualController(environment)
        pipeline = ClosedLoopPipeline(environment, processor, planner, policy, controller, run_dir, args.instruction)
        metrics = pipeline.run(args.seed if args.seed is not None else int(config.simulation["simulation"].get("seed", 0)), args.max_chunks)
        logging.getLogger(__name__).info("Run complete: %s", metrics)
        return 0
    finally:
        environment.close()


if __name__ == "__main__":
    raise SystemExit(main())