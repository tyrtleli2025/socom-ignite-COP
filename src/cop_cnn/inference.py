"""Inference utilities for deploying the COP CNN."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from PIL import Image
from torchvision import transforms

from .model import COPCNN


class COPInferenceEngine:
    """High-level interface for running inference on imagery tiles."""

    def __init__(
        self,
        weights_path: Path | str,
        class_names: Sequence[str],
        metadata_dim: int = 0,
        image_size: tuple[int, int] = (128, 128),
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = list(class_names)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
        self.model = COPCNN(num_classes=len(self.class_names), metadata_dim=metadata_dim)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        return self.transforms(image).unsqueeze(0).to(self.device)

    def _prepare_metadata(self, metadata: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if metadata is None:
            return None
        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)
        return metadata.to(self.device)

    def predict(self, image_path: Path | str, metadata: Optional[torch.Tensor] = None) -> dict:
        """Predict the mission label for a single image path."""

        with Image.open(image_path).convert("RGB") as image:
            image_tensor = self._prepare_image(image)
        metadata_tensor = self._prepare_metadata(metadata)
        with torch.no_grad():
            logits = self.model(image_tensor, metadata_tensor)
            probabilities = torch.softmax(logits, dim=1)
        top_prob, top_idx = probabilities.max(dim=1)
        return {
            "label": self.class_names[top_idx.item()],
            "confidence": top_prob.item(),
            "probabilities": {
                name: probabilities[0, idx].item() for idx, name in enumerate(self.class_names)
            },
        }

    def predict_batch(
        self, image_paths: Iterable[Path | str], metadata: Optional[Iterable[Optional[torch.Tensor]]] = None
    ) -> List[dict]:
        paths = list(image_paths)
        if metadata is None:
            metadata_list = [None] * len(paths)
        else:
            metadata_list = list(metadata)
            if len(metadata_list) != len(paths):
                raise ValueError("Metadata length must match number of image paths.")
        results: List[dict] = []
        for path, meta in zip(paths, metadata_list):
            results.append(self.predict(path, meta))
        return results


__all__ = ["COPInferenceEngine"]
