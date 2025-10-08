# Assignment 8: CIFAR-100 Classification with ResNet18

This assignment implements CIFAR-100 image classification using ResNet18 architecture from scratch.

## 📁 Structure

```
assignment8/
├── __init__.py                      # Module initialization
├── model.py                         # ResNet18 model for CIFAR-100 (100 classes)
├── data.py                          # CIFAR-100 data loading and augmentation
├── CIFAR100_ResNet18_Training.ipynb # Interactive training notebook
├── README.md                        # Complete project documentation
├── requirements.txt                 # Python dependencies
└── checkpoints/                     # Training checkpoints and outputs
```

## 🎯 Key Components

### `model.py`
- **ResNet18CIFAR100**: Full ResNet18 implementation with proper skip connections
- **BasicBlock**: Residual block with Sequential conv blocks and skip connections
- **Skip Connections**: Identity and downsample connections properly implemented
- **Comprehensive Dropout**: Dropout2d in every layer for regularization
- **Direct BasicBlock Usage**: No complex `_make_layer` function
- **11.2M parameters**, adapted for 32x32 images
- 100 output classes for CIFAR-100

### `data.py`
- **CIFAR100Dataset**: Custom dataset wrapper
- **Advanced augmentation** using Albumentations:
  - Pad + Random Crop
  - Horizontal Flip
  - Affine transformations
  - Brightness/Contrast adjustments
  - HSV color jitter
  - CoarseDropout (Cutout)
- **CIFAR-100 normalization** (proper mean/std)

## 🚀 Quick Start

### From Project Root

```bash
# Test the architecture
python main.py --test-architecture

# Train with default settings
python main.py

# Train with custom settings
python main.py --epochs 100 --batch_size 128 --scheduler cosine
```

### As a Module

```python
# Import from assignment8
from assignment8 import create_resnet18_cifar100, get_cifar100_data_loaders

# Create model
model = create_resnet18_cifar100(num_classes=100, dropout=0.1)

# Load data
train_loader, test_loader = get_cifar100_data_loaders(
    batch_size=128,
    augment=True
)
```

## 📊 Dataset Information

**CIFAR-100:**
- 100 classes (e.g., apple, aquarium_fish, baby, bear, etc.)
- 60,000 32×32 color images
- 50,000 training images (500 per class)
- 10,000 test images (100 per class)
- Mean: [0.5071, 0.4867, 0.4408]
- Std: [0.2675, 0.2565, 0.2761]

## 🏗️ Model Architecture

```
Input: (3, 32, 32)
├── Initial Conv: 3x3, 64 channels (32x32)
├── Layer 1: 2x BasicBlock, 64 channels (32x32)
├── Layer 2: 2x BasicBlock, 128 channels (16x16)
├── Layer 3: 2x BasicBlock, 256 channels (8x8)
├── Layer 4: 2x BasicBlock, 512 channels (4x4)
├── Global Average Pooling: (4x4) → (1x1)
├── Dropout: p=0.1
└── Fully Connected: 512 → 100 classes
```

**Features:**
- ✅ Residual connections
- ✅ Batch normalization
- ✅ Global Average Pooling
- ✅ Dropout regularization
- ✅ Proper weight initialization

## 📈 Expected Results

With default settings (OneCycleLR scheduler):
- **Train Accuracy**: 85-95%
- **Test Accuracy**: 70-75%
- **Training Time**: 30-60 minutes (on GPU)
- **Convergence**: 30-50 epochs
- **Scheduler**: OneCycleLR (faster convergence than step/cosine)

## 🔗 Dependencies

This module uses the **common** reusable infrastructure:
- `common.utils` - Device detection, plotting, checkpointing
- `common.trainer` - Generic training loop
- `common.config` - Configuration management

## 📝 Notes

- CIFAR-100 is significantly harder than CIFAR-10
- Data augmentation is crucial for good performance
- ResNet18 works well for this dataset size
- Training from scratch (no pre-trained weights)

## 📚 Documentation

- **README.md** - This file (complete project documentation)
- **common/README.md** - Documentation for reusable modules

---

For complete documentation, see the main project [README](../README.md)
