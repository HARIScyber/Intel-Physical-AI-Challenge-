"""Report availability of the project's Python dependencies."""
from __future__ import annotations

import importlib
import logging
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger(__name__)

DEPENDENCIES = {
    "PyTorch": "torch",
    "MuJoCo": "mujoco",
    "Transformers": "transformers",
    "Hugging Face Hub": "huggingface_hub",
    "LeRobot": "lerobot",
    "OpenVINO": "openvino",
    "OpenCV": "cv2",
    "NumPy": "numpy",
    "Gymnasium": "gymnasium",
    "PyYAML": "yaml",
}

def main() -> int:
    """Print a stable PASS/FAIL report and return non-zero when required items fail."""
    logging.basicConfig(level=logging.INFO)
    print("Physical AI Challenge setup check")
    print(f"Python: PASS ({platform.python_version()})")
    failures = 0
    for label, module_name in DEPENDENCIES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"{label}: PASS ({version})")
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            failures += 1
            print(f"{label}: FAIL ({exc})")
    print(f"Result: {'PASS' if failures == 0 else 'FAIL'} ({failures} missing or unavailable)")
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
