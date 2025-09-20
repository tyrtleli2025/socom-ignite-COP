"""Convolutional neural network backbone for common operating picture generation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Utility module composed of convolution, batch normalisation and activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - wrapper
        return self.block(x)


class COPCNN(nn.Module):
    """A lightweight CNN that fuses imagery with mission metadata."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3, metadata_dim: int = 0) -> None:
        super().__init__()
        self.metadata_dim = metadata_dim

        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=5, stride=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(128, 256),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        metadata_features = 0
        if metadata_dim > 0:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
            )
            metadata_features = 64
        else:
            self.metadata_encoder = None

        combined_features = 256 + metadata_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        visual_features = self.global_pool(self.stem(image)).flatten(1)
        if self.metadata_dim > 0 and metadata is not None:
            meta_features = self.metadata_encoder(metadata)
            fused = torch.cat([visual_features, meta_features], dim=1)
        else:
            fused = visual_features
        logits = self.classifier(fused)
        return logits

    def compute_loss(self, batch: dict, criterion: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute the training loss for a batch dictionary."""

        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        logits = self.forward(batch["image"], batch.get("metadata"))
        return criterion(logits, batch["label"])

    def freeze_backbone(self) -> None:
        """Freeze convolutional layers to fine-tune metadata fusion head."""

        for param in self.stem.parameters():
            param.requires_grad = False


__all__ = ["COPCNN", "ConvBlock"]
