# CIFAR-10 Classification Training Pipeline

A modular, generic training system for CIFAR-10 classification with the specified architecture requirements.

## 🏗️ Architecture Requirements

- **C1C2C3C40 Architecture**: No MaxPooling, convolutions with last one having stride=2
- **Depthwise Separable Convolution**: One layer uses depthwise separable convolution
- **Dilated Convolution**: One layer uses dilated convolution for increased receptive field
- **Global Average Pooling (GAP)**: Compulsory, with optional FC layer
- **Receptive Field**: > 44
- **Parameters**: < 200k
- **Target Accuracy**: 85%

## 📁 Modular Structure

### Core Modules

1. **`model.py`** - Model architecture and definitions
   - `CIFAR10Net`: Main model class with C1C2C3C40 architecture
   - `DepthwiseSeparableConv2d`: Custom depthwise separable convolution
   - `DilatedConv2d`: Custom dilated convolution
   - Architecture verification functions

2. **`data.py`** - Data loading and augmentation
   - CIFAR-10 dataset loading with albumentation transforms
   - Horizontal flip, ShiftScaleRotate, CoarseDropout augmentations
   - Generic data loading functions for other datasets

3. **`trainer.py`** - Generic training framework
   - `Trainer` class for model training
   - Generic training functions that work with any dataset
   - Checkpointing, early stopping, progress tracking

4. **`utils.py`** - Utility functions
   - Device detection, random seed setting
   - Model summary, checkpointing utilities
   - Training visualization and analysis

5. **`config.py`** - Configuration management
   - `TrainingConfig` dataclass for hyperparameters
   - Command-line argument parsing
   - Configuration validation and management

6. **`main.py`** - Entry point
   - Main training script with CLI support
   - Model setup, training orchestration
   - Results analysis and reporting

## 🚀 Usage

### Basic Training
```bash
python main.py --epochs 50 --batch_size 64 --lr 0.001
```

### With Custom Configuration
```bash
python main.py --epochs 30 --batch_size 128 --optimizer sgd --scheduler step
```

### Test Architecture Only
```bash
python main.py --test-architecture
```

### Available Options
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer type (adam, adamw, sgd)
- `--scheduler`: Scheduler type (step, cosine, plateau, cyclic)
- `--augment`: Enable/disable data augmentation
- `--target_accuracy`: Target accuracy to achieve (default: 85.0)

## 📊 Model Architecture

```
C1: 3x3 Conv (3→16) + BatchNorm + ReLU
C2: 3x3 Depthwise Separable Conv (16→32) + BatchNorm + ReLU
C3: 3x3 Dilated Conv (32→64, dilation=8) + BatchNorm + ReLU
C3b: 3x3 Dilated Conv (64→64, dilation=16) + BatchNorm + ReLU
C40: 3x3 Conv (64→128, stride=2) + BatchNorm + ReLU
GAP: Global Average Pooling
FC: Linear (128→10) + LogSoftmax
```

### Architecture Verification
- **Parameters**: 132,330 (< 200k ✅)
- **Receptive Field**: 55 (> 44 ✅)
- **Uses GAP**: ✅
- **Uses Depthwise Separable**: ✅
- **Uses Dilated Convolution**: ✅

## 🔧 Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## 📈 Data Augmentation

Uses albumentation library for:
- **Horizontal Flip**: 50% probability
- **ShiftScaleRotate**: Random shifts, scales, and rotations
- **CoarseDropout**: Random rectangular holes (16x16px)

## 🎯 Training Features

- **Early Stopping**: Prevents overfitting
- **Checkpointing**: Saves best and latest models
- **Learning Rate Scheduling**: Multiple scheduler options
- **Progress Tracking**: Real-time training progress
- **Visualization**: Training history plots
- **Modular Design**: Easy to adapt for other datasets

## 📁 Output Files

- `checkpoints/best_model.pth`: Best model checkpoint
- `checkpoints/latest_checkpoint.pth`: Latest checkpoint
- `training_history_*.png`: Training plots
- `*_training_info.json`: Training metadata

## 🔄 Generic Design

The training system is designed to be generic and work with other datasets:

1. **Model**: Replace `model.py` with your architecture
2. **Data**: Modify `data.py` for your dataset
3. **Config**: Update `config.py` for your hyperparameters
4. **Trainer**: Generic trainer works with any model/dataset

## 🧪 Testing

Test individual components:
```bash
python -c "from model import test_model_architecture; test_model_architecture()"
python -c "from data import test_data_loading; test_data_loading()"
python test_training.py
```

## 📋 Requirements Met

- ✅ C1C2C3C40 architecture (no MaxPooling)
- ✅ Depthwise Separable Convolution
- ✅ Dilated Convolution (dilation=8, 16)
- ✅ Global Average Pooling
- ✅ Receptive Field > 44 (55)
- ✅ Parameters < 200k (132,330)
- ✅ Albumentation augmentations
- ✅ Modular code structure
- ✅ Generic training framework
