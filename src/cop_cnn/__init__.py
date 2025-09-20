"""Top-level package for the SOCOM Ignite common operating picture CNN system."""

from .model import COPCNN
from .train import TrainingConfig, train_model
from .inference import COPInferenceEngine
from .data import COPDataset, generate_synthetic_samples

__all__ = [
    "COPCNN",
    "TrainingConfig",
    "train_model",
    "COPInferenceEngine",
    "COPDataset",
    "generate_synthetic_samples",
]
