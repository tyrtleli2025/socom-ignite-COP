"""Tests covering the CNN pipeline for the COP solution."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transforms = pytest.importorskip("torchvision.transforms")

from cop_cnn.data import COPDataset, generate_synthetic_samples, metadata_to_tensor
from cop_cnn.model import COPCNN
from cop_cnn.train import TrainingConfig, train_model


def test_dataset_and_model_forward(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate_synthetic_samples(data_dir, num_classes=2, samples_per_class=4, image_size=(32, 32), seed=1)

    dataset = COPDataset(
        data_dir,
        transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
        metadata_transform=metadata_to_tensor,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["metadata"].shape[-1] == 3

    model = COPCNN(num_classes=2, metadata_dim=3)
    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "metadata": torch.randn(2, 3),
        "label": torch.randint(0, 2, (2,)),
    }
    logits = model(batch["image"], batch["metadata"])
    assert logits.shape == (2, 2)
    loss = model.compute_loss(batch)
    assert loss.item() > 0


def test_training_loop_executes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate_synthetic_samples(data_dir, num_classes=3, samples_per_class=6, image_size=(32, 32), seed=2)

    config = TrainingConfig(
        data_dir=data_dir,
        num_classes=3,
        metadata_dim=3,
        image_size=(32, 32),
        batch_size=4,
        epochs=1,
        learning_rate=1e-3,
        validation_split=0.25,
        num_workers=0,
        output_dir=tmp_path / "artifacts",
        device="cpu",
        log_interval=1,
        seed=123,
    )

    model = train_model(config)
    assert isinstance(model, COPCNN)
    assert (config.output_dir / "best_model.pt").exists()
