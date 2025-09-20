"""Data ingestion utilities for the COP convolutional neural network system."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class COPSampleMetadata:
    """Structured metadata describing a single sample.

    Attributes
    ----------
    latitude: float
        Latitude of the sample centre in decimal degrees.
    longitude: float
        Longitude of the sample centre in decimal degrees.
    timestamp: datetime
        Timestamp associated with the intelligence collection.
    source: str
        Sensor modality identifier (e.g., "UAV-EO", "SAR").
    analyst_notes: Optional[str]
        Optional notes collected during manual exploitation.
    """

    latitude: float
    longitude: float
    timestamp: datetime
    source: str
    analyst_notes: Optional[str] = None


class COPDataset(Dataset):
    """Dataset representing COP tiles with metadata and mission labels.

    The dataset expects a directory structure where each label corresponds to a
    sub-folder. Metadata (geolocation, capture time, etc.) for each image can be
    optionally provided via a JSON side-car file with the same stem as the image.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        metadata_transform: Optional[Callable[[COPSampleMetadata], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist.")
        self.transform = transform
        self.metadata_transform = metadata_transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        self._scan()

    def _scan(self) -> None:
        """Scan the directory tree and populate samples."""

        for class_idx, class_dir in enumerate(sorted(p for p in self.root.iterdir() if p.is_dir())):
            self.class_to_idx[class_dir.name] = class_idx
            for image_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((image_path, class_idx))
            for image_path in sorted(class_dir.glob("*.png")):
                self.samples.append((image_path, class_idx))

        if not self.samples:
            raise RuntimeError(
                "No image samples were discovered. Ensure the dataset is organised as root/class_name/image.(jpg|png)."
            )

    def _load_metadata(self, image_path: Path) -> Optional[COPSampleMetadata]:
        meta_path = image_path.with_suffix(".json")
        if not meta_path.exists():
            return None
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        try:
            timestamp = datetime.fromisoformat(payload["timestamp"])
        except (KeyError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid timestamp metadata in {meta_path}.") from exc
        return COPSampleMetadata(
            latitude=float(payload.get("latitude", 0.0)),
            longitude=float(payload.get("longitude", 0.0)),
            timestamp=timestamp,
            source=payload.get("source", "unknown"),
            analyst_notes=payload.get("analyst_notes"),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, label_idx = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:  # pragma: no cover - executed in production but not tests
            image_tensor = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())).float())
            image_tensor = image_tensor.view(image.size[1], image.size[0], 3).permute(2, 0, 1) / 255.0

        metadata = self._load_metadata(image_path)
        metadata_tensor: Optional[torch.Tensor] = None
        if metadata and self.metadata_transform:
            metadata_tensor = self.metadata_transform(metadata)
        elif metadata:
            metadata_tensor = torch.tensor(
                [metadata.latitude, metadata.longitude], dtype=torch.float32
            )  # simple default encoding

        return {
            "image": image_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "metadata": metadata_tensor,
            "path": str(image_path),
        }


def generate_synthetic_samples(
    destination: Path | str,
    num_classes: int = 3,
    samples_per_class: int = 20,
    image_size: Tuple[int, int] = (64, 64),
    seed: Optional[int] = 17,
) -> None:
    """Generate a synthetic dataset composed of abstract shapes.

    This utility enables rapid experimentation when real mission imagery is not
    accessible in the development environment. The generated dataset supports the
    same directory layout expected by :class:`COPDataset`.
    """

    rng = random.Random(seed)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)

    for class_idx in range(num_classes):
        class_dir = destination_path / f"class_{class_idx}"
        class_dir.mkdir(exist_ok=True)
        for sample_idx in range(samples_per_class):
            image = Image.new("RGB", image_size, color=(0, 0, 0))
            pixels = image.load()
            centre_x = rng.randint(10, image_size[0] - 10)
            centre_y = rng.randint(10, image_size[1] - 10)
            radius = rng.randint(5, min(image_size) // 2)
            color = (
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
            )
            for x in range(image_size[0]):
                for y in range(image_size[1]):
                    if (x - centre_x) ** 2 + (y - centre_y) ** 2 <= radius**2:
                        pixels[x, y] = color
            image_path = class_dir / f"sample_{sample_idx}.png"
            image.save(image_path)

            metadata = {
                "latitude": rng.uniform(-90, 90),
                "longitude": rng.uniform(-180, 180),
                "timestamp": datetime.utcnow().isoformat(),
                "source": rng.choice(["UAV-EO", "UAV-IR", "SAT-SAR"]),
                "analyst_notes": "Synthetic sample",
            }
            with image_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle)


def metadata_to_tensor(metadata: COPSampleMetadata) -> torch.Tensor:
    """Encode metadata into a numeric tensor suitable for model conditioning."""

    timestamp_feature = metadata.timestamp.timestamp() / (60 * 60 * 24)  # days since epoch
    return torch.tensor(
        [metadata.latitude, metadata.longitude, timestamp_feature], dtype=torch.float32
    )


__all__ = [
    "COPDataset",
    "COPSampleMetadata",
    "generate_synthetic_samples",
    "metadata_to_tensor",
]
