"""Convert local scripted episodes with the installed LeRobot writer API."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def _lerobot_dataset_class() -> Any:
    """Import the current LeRobot dataset class, including the editable checkout."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except (ModuleNotFoundError, SyntaxError) as first_error:
        source_root = ROOT / "lerobot" / "src"
        if not source_root.is_dir():
            raise RuntimeError("LeRobot is not importable and no local editable checkout was found") from first_error
        sys.path.insert(0, str(source_root))
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            return LeRobotDataset
        except SyntaxError as exc:
            raise RuntimeError("This LeRobot checkout requires Python 3.12 or newer; run conversion with Python 3.12+") from exc


def convert(source: str | Path, destination: str | Path, config: dict[str, Any] | None = None) -> Path:
    """Convert raw episodes using create, add_frame, save_episode, and finalize."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    episodes = sorted(item for item in source_path.iterdir() if item.is_dir() and item.name.startswith("episode_"))
    if not episodes:
        raise ValueError(f"No raw episodes found in {source_path}")
    dataset_class = _lerobot_dataset_class()
    first_metadata = json.loads((episodes[0] / "metadata.json").read_text(encoding="utf-8"))
    data = np.load(episodes[0] / "data.npz")
    state_shape = tuple(int(value) for value in data["observation_state"].shape[1:])
    action_shape = tuple(int(value) for value in data["action"].shape[1:])
    frame = cv2.imread(str(episodes[0] / first_metadata["image_paths"][0]["overhead"]), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The first raw overhead image could not be decoded")
    height, width = frame.shape[:2]
    options = config or {}
    features = {
        "observation.images.overhead": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
        "observation.images.front": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": state_shape, "names": None},
        "action": {"dtype": "float32", "shape": action_shape, "names": None},
    }
    destination_path.mkdir(parents=True, exist_ok=True)
    if any(destination_path.iterdir()):
        raise FileExistsError(f"LeRobot destination must be empty: {destination_path}")
    dataset = dataset_class.create(repo_id=str(options.get("repo_id", "local/dinner_table")), fps=int(options.get("fps", 20)), features=features, root=destination_path, robot_type="so101_simulation", use_videos=False)
    try:
        for episode in episodes:
            _append_episode(dataset, episode)
        dataset.finalize()
    except Exception:
        dataset.clear_episode_buffer()
        raise
    LOGGER.info("Converted %d episodes to %s", len(episodes), destination_path)
    return destination_path


def _append_episode(dataset: Any, episode: Path) -> None:
    """Append one raw episode and preserve its language task annotation."""
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    data = np.load(episode / "data.npz")
    states = np.asarray(data["observation_state"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    timestamps = np.asarray(data["timestamp"], dtype=np.float64)
    image_paths = metadata["image_paths"]
    if not (len(states) == len(actions) == len(timestamps) == len(image_paths)):
        raise ValueError(f"Raw episode has inconsistent lengths: {episode}")
    task = str(metadata["language_instruction"])
    for index in range(len(states)):
        images: dict[str, np.ndarray] = {}
        for camera_name, feature_name in (("overhead", "observation.images.overhead"), ("front", "observation.images.front")):
            image = cv2.imread(str(episode / image_paths[index][camera_name]), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3:
                raise ValueError(f"Invalid image at {episode / image_paths[index][camera_name]}")
            images[feature_name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dataset.add_frame({**images, "observation.state": states[index], "action": actions[index], "task": task})
    dataset.save_episode()


def main() -> int:
    """Convert raw demonstrations to a local LeRobot dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="datasets/raw")
    parser.add_argument("destination", nargs="?", default="datasets/lerobot")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    convert(args.source, args.destination)
    print(f"LeRobot dataset written locally: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())