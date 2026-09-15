"""Training entry point, intentionally disabled until data and policy are configured."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physical_ai.config import load_config
from policy.training import train

if __name__ == "__main__":
    train(load_config().training)
