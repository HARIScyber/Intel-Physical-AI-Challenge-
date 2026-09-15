"""YAML configuration loading for the simulation-first project."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project settings with path resolution relative to the repository."""

    project_root: Path
    simulation: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    openvino: dict[str, Any] = field(default_factory=dict)

    def path(self, value: str | Path) -> Path:
        """Resolve a configured relative path without relying on machine-specific paths."""
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.project_root / candidate


def load_config(config_dir: str | Path = "configs", project_root: str | Path | None = None) -> ProjectConfig:
    """Load the four project YAML files and raise actionable errors on invalid input."""
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    directory = Path(config_dir)
    if not directory.is_absolute():
        directory = root / directory
    try:
        documents: dict[str, dict[str, Any]] = {}
        for name in ("simulation", "training", "evaluation", "openvino"):
            path = directory / f"{name}.yaml"
            if not path.is_file():
                raise FileNotFoundError(f"Missing configuration file: {path}")
            with path.open("r", encoding="utf-8") as handle:
                documents[name] = yaml.safe_load(handle) or {}
        return ProjectConfig(root, **documents)
    except (OSError, yaml.YAMLError) as exc:
        LOGGER.exception("Unable to load configuration from %s", directory)
        raise RuntimeError(f"Configuration loading failed: {exc}") from exc
