"""Benchmark the local Python execution boundary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.benchmark import benchmark

if __name__ == "__main__":
    print(benchmark(lambda: None))
