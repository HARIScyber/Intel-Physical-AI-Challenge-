"""Evaluation entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physical_ai.config import load_config
from evaluation.evaluate import evaluate

if __name__ == "__main__":
    print(evaluate(load_config().evaluation))
