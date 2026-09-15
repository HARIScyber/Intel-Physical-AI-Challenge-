# Setup

Use Python 3.10 or newer. Create a virtual environment and install `requirements.txt`. `scripts/setup_check.py` reports every package individually. A failed check means the associated integration is unavailable; the simulation requires MuJoCo and Gymnasium, while model, vision, and optimization packages are needed only for their phases.

No absolute machine paths are stored in configuration. Paths are resolved relative to the repository root.
