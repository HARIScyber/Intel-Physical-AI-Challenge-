# Bimanual VLA Manipulation with Multi-Modal Reasoning

Software-first Intel Physical AI Challenge foundation for setting a dinner table with two simulated SO-101 arms. MuJoCo is the first execution target; real hardware is not required.

## Quick start

```powershell
cd d:\physical-ai-challenge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/setup_check.py
python scripts/run_simulation.py
```

The simulation is headless and runs from `configs/simulation.yaml`. Override its duration for a smoke test with `python scripts/run_simulation.py --duration 0.1`.

## Architecture

YAML files are loaded by `physical_ai.config.load_config` into one typed `ProjectConfig`. The MuJoCo scene is built by `simulation.scene`, while `simulation.mujoco_env.BimanualMujocoEnv` owns reset and stepping. Perception produces `SceneState`; language normalizes commands; planning produces an `ActionSequence`; a model-independent policy interface feeds the control layer. LeRobot, Transformers, and OpenVINO adapters are deliberately explicit integration boundaries until a real checkpoint and dataset are selected.

## Current scope

The foundation executes a deterministic MuJoCo scene and exposes typed seams for future perception, planning, policy, evaluation, and Intel Core Ultra benchmarking. VLA training, model conversion, and demonstration collection are not implemented yet and fail explicitly rather than pretending to be complete.

## Development checks

```powershell
python -m compileall -q physical_ai simulation perception language planning policy control evaluation optimization datasets scripts
python -m unittest discover -s tests
```

See `docs/setup.md` and `docs/architecture.md` for the next implementation stages.
