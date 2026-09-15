# Architecture

The system is organized as a pipeline:

1. MuJoCo produces simulated state and camera observations.
2. Perception turns frames into detections, masks, and `SceneState`.
3. Language turns instructions into `TaskCommand`.
4. Planning creates collision-checked bimanual `ActionSequence` objects.
5. A `BasePolicy` implementation predicts actions.
6. Controllers dispatch synchronized commands to both simulated arms.
7. Evaluation records success, latency, and robustness by seed.
8. Optimization owns OpenVINO conversion and Intel device benchmarks.

Configuration is loaded once through `physical_ai.config` and passed to each boundary.
