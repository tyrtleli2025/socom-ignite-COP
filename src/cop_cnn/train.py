"""Training utilities for the COP CNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .data import COPDataset, metadata_to_tensor
from .model import COPCNN


@dataclass
class TrainingConfig:
    """Configuration dataclass controlling model training."""

    data_dir: Path
    num_classes: int
    metadata_dim: int = 0
    image_size: Tuple[int, int] = (128, 128)
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.2
    num_workers: int = 0
    output_dir: Path = Path("artifacts")
    device: Optional[str] = None
    log_interval: int = 10
    seed: int = 42

    def resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_transforms(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ]
    )


def _prepare_dataloaders(config: TrainingConfig) -> tuple[DataLoader, Optional[DataLoader]]:
    transform = _build_transforms(config.image_size)
    dataset = COPDataset(
        config.data_dir,
        transform=transform,
        metadata_transform=metadata_to_tensor if config.metadata_dim > 0 else None,
    )
    val_loader: Optional[DataLoader] = None
    if 0 < config.validation_split < 1:
        val_size = int(len(dataset) * config.validation_split)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    return train_loader, val_loader


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    batch = batch.copy()
    batch["image"] = batch["image"].to(device)
    batch["label"] = batch["label"].to(device)
    if batch.get("metadata") is not None:
        batch["metadata"] = batch["metadata"].to(device)
    return batch


def _evaluate(model: COPCNN, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            logits = model(batch["image"], batch.get("metadata"))
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch["label"]).sum().item()
            total += batch["label"].numel()
    return correct / max(total, 1)


def train_model(config: TrainingConfig) -> COPCNN:
    """Train a :class:`COPCNN` using the provided configuration."""

    torch.manual_seed(config.seed)
    device = config.resolve_device()
    train_loader, val_loader = _prepare_dataloaders(config)

    model = COPCNN(num_classes=config.num_classes, metadata_dim=config.metadata_dim)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch_to_device(batch, device)
            loss = model.compute_loss(batch, criterion)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            if step % config.log_interval == 0:
                avg_loss = running_loss / config.log_interval
                print(f"Epoch {epoch}/{config.epochs} step {step}: loss={avg_loss:.4f}")
                running_loss = 0.0

        scheduler.step()

        if val_loader is not None:
            val_accuracy = _evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: validation accuracy={val_accuracy:.3f}")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), config.output_dir / "best_model.pt")
        else:
            torch.save(model.state_dict(), config.output_dir / f"model_epoch_{epoch}.pt")

    return model


__all__ = ["TrainingConfig", "train_model"]
