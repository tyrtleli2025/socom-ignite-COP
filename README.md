# SOCOM Ignite Common Operating Picture CNN

This repository contains a reference implementation of an AI-enabled common operating picture (COP) generator driven by a convolutional neural network (CNN). The system is designed to ingest heterogeneous imagery, fuse mission metadata, and deliver actionable classifications that accelerate staff decision cycles and reduce manual triage effort.

## Key Capabilities

- **Multi-source data integration** – Unified dataset abstraction handles imagery tiles alongside geospatial/temporal metadata to support AI-enabled COP workflows.
- **Mission-ready CNN architecture** – Lightweight convolutional backbone fuses visual features and mission metadata for responsive inference in tactical environments.
- **End-to-end pipeline** – Utilities for synthetic data generation, training, validation, and deployment-ready inference.
- **Human-centric insights** – Metadata fusion and configurable class labels enable alignment with human analyst workflows and mission semantics.

## Getting Started

### Prerequisites

Install dependencies into your Python environment (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

### Generate a Synthetic Dataset

Synthetic samples emulate UAV or satellite captures and allow rapid experimentation without sensitive data:

```python
from pathlib import Path
from cop_cnn.data import generate_synthetic_samples

generate_synthetic_samples(Path("dataset"), num_classes=4, samples_per_class=50)
```

The generator produces the directory layout expected by the dataset loader:

```
dataset/
  class_0/
    sample_0.png
    sample_0.json
  class_1/
  ...
```

### Train the Model

Use the training configuration dataclass to tailor the pipeline for your mission dataset. The example below trains for a single epoch on the synthetic data:

```python
from pathlib import Path
from cop_cnn.train import TrainingConfig, train_model

config = TrainingConfig(
    data_dir=Path("dataset"),
    num_classes=4,
    metadata_dim=3,
    image_size=(128, 128),
    batch_size=16,
    epochs=5,
)

model = train_model(config)
```

Model checkpoints are stored in `artifacts/` by default and include the highest performing validation model.

### Run Inference

Load trained weights and execute predictions against new imagery tiles:

```python
import torch
from cop_cnn.inference import COPInferenceEngine
from cop_cnn.data import metadata_to_tensor, COPSampleMetadata
from datetime import datetime

engine = COPInferenceEngine(
    weights_path="artifacts/best_model.pt",
    class_names=["safe", "needs_review", "priority"],
    metadata_dim=3,
)

metadata = metadata_to_tensor(
    COPSampleMetadata(
        latitude=30.1,
        longitude=45.2,
        timestamp=datetime.utcnow(),
        source="UAV-EO",
    )
)

result = engine.predict("path/to/image.png", metadata=metadata)
print(result)
```

## Testing

Run the automated tests to validate the training and inference stack:

```bash
pytest
```

## Project Structure

```
src/cop_cnn/
  __init__.py          # Package exports
  data.py              # Dataset, metadata handling, synthetic generator
  model.py             # CNN architecture and utilities
  train.py             # Training loop and configuration
  inference.py         # Model loading and inference helpers
requirements.txt       # Core dependencies
README.md              # Project overview and usage
```

## Extending the System

- Integrate operational data feeds by subclassing `COPDataset` or enriching metadata encodings.
- Add explainability modules (Grad-CAM, saliency maps) to support analyst trust.
- Deploy the inference engine inside edge compute nodes for low-latency COP updates.

## License

This project is released under the MIT license.
