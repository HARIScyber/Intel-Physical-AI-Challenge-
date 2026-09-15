# Training

Training is intentionally deferred. The planned order is: collect successful MuJoCo demonstrations, validate them, convert through the installed LeRobot dataset API, select a compatible policy checkpoint, then add a reproducible training command.

The deterministic baseline is available with `python scripts/run_demo.py --headless`. It exercises the semantic drawer, grasp, transport, placement, coordination, handoff, and verification phases without a VLA model. Grasp success is counted only when MuJoCo reports contact between an object and gripper geometry.

## Optional VLA inference

Policy selection is configured in `configs/training.yaml`:

```yaml
policy:
	type: "smolvla"
	pretrained: true
	checkpoint: null
```

The default `null` checkpoint performs no model loading. `policy.create_policy()` returns the isolated `LeRobotVLA` adapter for SmolVLA, while `ScriptedPolicy` and `DummyPolicy` remain available for local simulation and tests. When a checkpoint is configured, the adapter uses LeRobot's documented `make_policy_config`, `make_policy`, `make_pre_post_processors`, and `select_action` interfaces. It converts camera images, robot state, and language into the dual-arm action mapping.

Model loading is lazy and failures are explicit. Set `fallback_to_scripted: true` to keep a simulation rollout running when the checkpoint or optional LeRobot runtime is unavailable. The current checkout's LeRobot source requires Python 3.12 or newer.
