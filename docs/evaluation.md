# Evaluation

Evaluation should run fixed seeds from `configs/evaluation.yaml`, report task success and per-episode latency, and include perturbations such as object pose, lighting, and instruction changes.

Domain-randomized episodes support `easy`, `medium`, and `hard` modes. Every MuJoCo reset returns the seed, mode, and applied randomization record. Use the same seed and mode to replay a failure:

```python
observation, info = env.reset(seed=101, options={"evaluation_mode": "hard"})
print(info["randomization"])
```

Demonstration generation can vary both physics and language wording:

```powershell
python datasets/generate_demonstrations.py --episodes 3 --mode hard
```
