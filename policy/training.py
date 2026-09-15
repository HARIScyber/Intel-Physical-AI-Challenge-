"""Training boundary; no training is enabled in the foundation release."""
import logging

LOGGER = logging.getLogger(__name__)

def train(config: dict) -> None:
    """Reject accidental training until dataset and policy selection are configured."""
    if not config:
        raise ValueError("training configuration is required")
    raise NotImplementedError("VLA training is intentionally deferred")
