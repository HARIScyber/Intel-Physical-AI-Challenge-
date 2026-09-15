"""Checkpoint configuration and validation for optional policy loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointReference:
    """Local path or Hub identifier for a policy checkpoint."""

    value: str

    @property
    def is_local(self) -> bool:
        """Return whether the reference resolves to an existing local path."""
        return Path(self.value).exists()


def validate_checkpoint(path: str | Path) -> Path:
    """Validate a required local checkpoint path."""
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def checkpoint_reference(value: str | Path | None) -> CheckpointReference | None:
    """Normalize an optional checkpoint without requiring it to exist."""
    return None if value in (None, "") else CheckpointReference(str(value))
