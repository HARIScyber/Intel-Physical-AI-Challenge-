# OpenVINO Benchmark Report

Generated: `2026-09-14T09:47:10.243897+00:00`

## Methodology

- Checkpoint: `lerobot/smolvla_base`
- Device requested: `cpu`
- Identical sample seed: `123`
- Sample SHA-256: `7d9463da7d8937f64a7140f3a3d88f1fd7e46557f19ea02ac24a473e0fe8d0c7`
- Warmup iterations: `0`
- Measured inference iterations: `1`
- Task episodes: `1`
- Task steps per episode: `1`

The benchmark uses one fixed image, mask, token, state, action-noise, and timestep sample for every backend. The exported graph is the same SmolVLA forward computation with KV-cache reuse disabled so the PyTorch and OpenVINO paths receive identical tensors.

The task benchmark executes one bounded dual-arm action in the MuJoCo environment per episode. Task success means a finite model action was executed without a collision; `environment_task_complete` records the simulator's full dinner-table predicate separately.

## Inference Results

| Backend | Precision | Device | Load (s) | Warmup (s) | Single (s) | Avg (s) | p50 (s) | p95 (s) | Throughput (samples/s) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| PyTorch | fp32 | cpu | 35.8648 | 0.0000 | 2.5819 | 3.7346 | 3.7346 | 3.7346 | 0.27 | ready |
| OpenVINO | fp32 | CPU | 26.6185 | 0.0004 | 129.5420 | 133.7905 | 133.7905 | 133.7905 | 0.01 | ready |
| OpenVINO | fp16 | CPU | 18.8097 | 0.0001 | 137.0446 | 113.8637 | 113.8637 | 113.8637 | 0.01 | ready |
| OpenVINO | int8 | CPU | N/A | N/A | N/A | N/A | N/A | N/A | N/A | unsupported |

## End-to-End Task Results

| Backend | Precision | Task Success Rate | Avg Episode Duration (s) | Avg Action Latency (s) | Episodes | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| PyTorch | fp32 | 1.00 | 4.0309 | 2.4160 | 1 | ready |
| OpenVINO | fp32 | 1.00 | 138.5557 | 137.1687 | 1 | ready |
| OpenVINO | fp16 | 1.00 | 120.0578 | 118.5600 | 1 | ready |
| OpenVINO | int8 | N/A | N/A | N/A | 1 | unsupported |

## Precision Artifacts

- **FP32**: `ready` — OpenVINO IR exported from the identical-input forward graph (`3316974` bytes).
- **FP16**: `ready` — FP16-compressed OpenVINO IR (`3316974` bytes).
- **INT8**: `unsupported` — NNCF is not installed; calibrated INT8 post-training quantization was not run (`None` bytes).

## Analysis

- OpenVINO FP32 average latency is 0.03x the PyTorch baseline on `CPU`.
- OpenVINO FP16 average latency is 0.03x the PyTorch baseline on `CPU`.
- CPU precision hint for FP16: `<Type: 'float32'>`; FP16 may be emulated when the CPU reports only FP32 inference precision.

## Reproduction

```bash
python optimization/intel_benchmark.py lerobot/smolvla_base --iterations 3 --warmup 1 --task-episodes 1 --task-steps 1
```
