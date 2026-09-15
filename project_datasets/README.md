# Datasets

This directory is reserved for generated demonstrations and LeRobot-compatible exports. Generated data is intentionally excluded from Git. Run the validators before training.

## Generate local demonstrations

```powershell
python datasets/generate_demonstrations.py --episodes 3
python datasets/validate_dataset.py datasets/raw
```

Each `episode_NNNN` directory contains:

- `frames/frame_*_overhead.png` and `frames/frame_*_front.png`
- `data.npz` with `observation_state`, `action`, and `timestamp`
- `metadata.json` with episode ID, seed, language instruction, task schema, action names, and frame paths
- `perception_debug/` RGB, detection, and segmentation diagnostics

The generator uses the deterministic scripted baseline and only local MuJoCo, OpenCV, NumPy, and YAML configuration. No cloud service is required.

## Convert to LeRobot

The converter uses the installed checkout's `LeRobotDataset.create`, `add_frame`, `save_episode`, and `finalize` APIs. It does not upload to the Hub.

```powershell
python datasets/convert_to_lerobot.py datasets/raw datasets/lerobot
python datasets/validate_dataset.py --lerobot datasets/lerobot
```

Inspect the local dataset without uploading:

```powershell
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset('local/dinner_table', root='datasets/lerobot', download_videos=False); print(d.features); print(d.num_episodes, d.num_frames); print(d[0].keys())"
```

The current LeRobot checkout in this workspace uses Python 3.12 syntax. Use a Python 3.12+ interpreter for conversion and validation; Python 3.11 will fail before LeRobot can be imported.
