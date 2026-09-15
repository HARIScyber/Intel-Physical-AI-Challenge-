# Simulation

`python scripts/run_simulation.py` loads YAML, creates a self-contained MuJoCo XML scene, resets it, steps physics, and closes native state. The initial scene contains a table, plate, cup, and two arm-base markers. Robot geometry and actuators should be replaced with the validated SO-101 MuJoCo model in the next phase.
