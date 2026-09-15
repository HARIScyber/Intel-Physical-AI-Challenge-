"""Validate raw scripted demonstration episodes without cloud services."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np


def validate_lerobot(path: str | Path) -> bool:
    """Load a local LeRobot dataset and validate samples, images, actions, and episodes."""
    root = Path(path)
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot metadata not found under {root / 'meta'}")
    from datasets.convert_to_lerobot import _lerobot_dataset_class
    dataset_class = _lerobot_dataset_class()
    try:
        dataset = dataset_class("local/dinner_table", root=root, download_videos=False)
    except Exception as exc:
        raise ValueError(f"LeRobot dataset could not be loaded: {exc}") from exc
    required_features = {"observation.images.overhead", "observation.images.front", "observation.state", "action"}
    missing = required_features - set(dataset.features)
    if missing:
        raise ValueError(f"LeRobot dataset missing features: {sorted(missing)}")
    if dataset.num_episodes < 1 or dataset.num_frames < 1:
        raise ValueError("LeRobot dataset contains no episodes or frames")
    expected_state = tuple(dataset.features["observation.state"]["shape"])
    expected_action = tuple(dataset.features["action"]["shape"])
    for index in range(min(dataset.num_frames, 10)):
        sample = dataset[index]
        for image_key in ("observation.images.overhead", "observation.images.front"):
            image = sample[image_key]
            if image is None or getattr(image, "numel", lambda: np.asarray(image).size)() == 0:
                raise ValueError(f"Invalid decoded image at frame {index}: {image_key}")
        state = np.asarray(sample["observation.state"])
        action = np.asarray(sample["action"])
        if state.shape != expected_state:
            raise ValueError(f"Invalid state shape at frame {index}: {state.shape} != {expected_state}")
        if action.shape != expected_action:
            raise ValueError(f"Invalid action shape at frame {index}: {action.shape} != {expected_action}")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            raise ValueError(f"NaN or infinite value at frame {index}")
    if len(dataset.meta.episodes) != dataset.num_episodes:
        raise ValueError("Episode metadata count does not match loaded dataset")
    return True

def validate(path: str | Path, config: dict[str, Any] | None = None) -> bool:
    """Validate every episode and raise ``ValueError`` with all discovered issues."""
    del config
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    issues: list[str] = []
    episodes = sorted(item for item in root.iterdir() if item.is_dir() and item.name.startswith("episode_"))
    if not episodes:
        raise ValueError(f"no episode directories found under {root}")
    for episode in episodes:
        issues.extend(_validate_episode(episode))
    if issues:
        raise ValueError("dataset validation failed:\n" + "\n".join(issues))
    return True

def _validate_episode(episode: Path) -> list[str]:
    """Validate metadata, arrays, frames, shapes, bounds, and timestamps."""
    issues: list[str] = []
    metadata_path = episode / "metadata.json"
    data_path = episode / "data.npz"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        data = np.load(data_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"{episode.name}: cannot load episode: {exc}"]
    required = {"episode_id", "language_instruction", "task", "step_count", "image_paths"}
    missing = required - set(metadata)
    if missing:
        issues.append(f"{episode.name}: missing metadata keys {sorted(missing)}")
    states = np.asarray(data["observation_state"])
    actions = np.asarray(data["action"])
    timestamps = np.asarray(data["timestamp"])
    if states.ndim != 2 or states.shape[1] != 67:
        issues.append(f"{episode.name}: incorrect observation_state shape {states.shape}, expected (N, 67)")
    if actions.ndim != 2 or actions.shape[1] != 15:
        issues.append(f"{episode.name}: incorrect action shape {actions.shape}, expected (N, 15)")
    if len(states) != len(actions) or len(states) != len(timestamps) or len(states) != len(metadata.get("image_paths", [])):
        issues.append(f"{episode.name}: episode arrays and metadata frame counts disagree")
    if int(metadata.get("step_count", -1)) != len(states):
        issues.append(f"{episode.name}: incorrect episode length metadata")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)) or not np.all(np.isfinite(timestamps)):
        issues.append(f"{episode.name}: NaN or infinite numeric value")
    if len(timestamps) and (timestamps[0] < 0 or np.any(np.diff(timestamps) < 0)):
        issues.append(f"{episode.name}: invalid timestamps")
    if len(actions) and (np.any(actions[:, :12] < -3.15) or np.any(actions[:, :12] > 3.15) or np.any(actions[:, 12:14] < 0) or np.any(actions[:, 12:14] > 1) or np.any(actions[:, 14] < -0.35) or np.any(actions[:, 14] > 0.02)):
        issues.append(f"{episode.name}: action bounds exceeded")
    for index, frame_set in enumerate(metadata.get("image_paths", [])):
        for camera_name in metadata.get("camera_names", []):
            frame_path = episode / frame_set.get(camera_name, "")
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
                issues.append(f"{episode.name}: invalid or missing {camera_name} frame at step {index}")
    return issues

def main() -> int:
    """Run dataset validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="datasets/raw")
    parser.add_argument("--lerobot", action="store_true", help="Validate a local LeRobot dataset")
    args = parser.parse_args()
    if args.lerobot:
        validate_lerobot(args.path)
    else:
        validate(args.path)
    print(f"Dataset valid: {args.path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())