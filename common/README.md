# Common Reusable Training Infrastructure

This directory contains reusable training infrastructure that can be shared across different assignments and datasets.

## 📦 Modules

### `utils.py`
Generic utility functions for training:
- Device detection (CUDA/MPS/CPU)
- Random seed setting
- Model summary printing
- Checkpoint save/load
- Training history plotting
- Time formatting
- Accuracy calculation

### `trainer.py`
Generic training module:
- `Trainer` class for any PyTorch model
- Epoch training/testing
- Early stopping
- Checkpoint management
- Training history tracking
- Support for different schedulers

### `config.py`
Configuration management:
- `TrainingConfig` dataclass
- Command-line argument parsing
- JSON save/load
- Configuration validation
- Default settings

## 🔧 Usage

### Import the entire module
```python
from common import (
    get_device, set_random_seed, Trainer,
    TrainingConfig, plot_training_history
)
```

### Or import specific modules
```python
from common.utils import get_device, set_random_seed
from common.trainer import Trainer
from common.config import TrainingConfig
```

## 📚 Example

```python
from common import get_device, set_random_seed, create_trainer, TrainingConfig

# Setup
config = TrainingConfig(epochs=50, batch_size=128)
device = get_device()
set_random_seed(42)

# Create trainer
trainer = create_trainer(
    model=your_model,
    device=device,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    config=config.to_dict()
)

# Train
history = trainer.train(
    num_epochs=config.epochs,
    checkpoint_dir='./checkpoints'
)
```

## 🎯 Design Philosophy

These modules are designed to be:
- **Dataset-agnostic**: Work with any dataset
- **Model-agnostic**: Work with any PyTorch model
- **Flexible**: Support multiple optimizers and schedulers
- **Reusable**: No duplicate code across assignments
- **Well-tested**: Verified to work correctly

## 📝 Version

Current version: 1.0.0

