# MNIST Classification: CNN Architecture Comparison

## Technical Overview
Implementation of three CNN variants for MNIST digit classification, comparing the effects of dropout placement and batch normalization on model convergence and performance.

## Specifications
- **Dataset**: MNIST (60k train, 10k test, 28×28 grayscale)
- **Parameter Constraint**: <20k parameters
- **Target Accuracy**: >99.4% within 20 epochs
- **Framework**: PyTorch 2.8.0
- **Hardware**: MPS

## Data Pipeline
- **Augmentation**: RandomRotation(5°), RandomAffine, ColorJitter
- **Normalization**: mean=0.1307, std=0.3081
- **Batch Size**: 32
- **Workers**: 2 (parallel loading)
## Architecture Specifications

### Base Architecture: SmallMNISTNet4Block
- **Parameters**: 17,938 (under 20k constraint)
- **Structure**: 4 conv blocks → Global Average Pooling → Linear(32→10)
- **Activations**: ReLU
- **Pooling**: MaxPool2d(2) after each block
- **Padding**: Same padding (kernel_size=3, padding=1)

### Architecture Variants

| Model | Block 3 Dropout | Block 4 Dropout | Batch Normalization | Parameters |
|-------|----------------|-----------------|-------------------|------------|
| **v1** | Dropout2d(0.1) | Dropout2d(0.1) | None | 17,938 |
| **v2** | None | None | None | 17,938 |
| **Final** | Dropout2d(0.1) | None | All conv layers | 18,434 |

### Training Hyperparameters
- **Optimizer**: Adam (lr=0.001, weight_decay=0)
- **Loss**: CrossEntropyLoss
- **Scheduler**: StepLR (step_size=7, gamma=0.1)
- **Epochs**: 20
- **Batch Size**: 32
- **Early Stopping**: patience=10, min_delta=0.001

## Performance Results

### Test Accuracy Comparison

| Model | Final Accuracy | Epochs to 99%+ | Convergence Rate |
|-------|----------------|----------------|------------------|
| **<span style="color:red">Final</span>** | **<span style="color:red">99.55%</span>** | **<span style="color:red">5</span>** | **<span style="color:red">1.2% per epoch</span>** |
| **v2** | 99.38% | 8 | 0.9% per epoch |
| **v1** | 99.36% | 8 | 0.9% per epoch |

### Convergence Analysis

#### Model v1: Full Dropout
- **Dropout**: All 4 blocks (rate=0.1)
- **Epoch 1**: Train=70.51%, Test=93.59%
- **Epoch 8**: First 99%+ (99.13%)
- **Final**: 99.36% (Epoch 20)

#### Model v2: Reduced Dropout
- **Dropout**: Blocks 1-2 only
- **Epoch 1**: Train=77.69%, Test=96.68%
- **Epoch 8**: First 99%+ (99.20%)
- **Final**: 99.38% (Epoch 20)

#### Model Final: Batch Normalization
- **BN**: All conv layers (momentum=0.1)
- **Dropout**: Blocks 1-2 only
- **Epoch 1**: Train=90.31%, Test=98.76%
- **Epoch 5**: First 99%+ (99.18%)
- **Final**: 99.55% (Epoch 20)

## Technical Analysis

### Key Findings
1. **Dropout Impact**: Final layer dropout reduces convergence speed by 3 epochs
2. **Batch Normalization**: Provides 3 epochs faster convergence and 0.17% accuracy improvement
3. **Feature Preservation**: Removing dropout from high-level layers improves learning efficiency

### Implementation Details
- **Device**: MPS (Apple Silicon)
- **Memory**: ~0.46MB forward pass, 0.07MB parameters
- **Training Time**: ~2-3 minutes per model (20 epochs)
- **Reproducibility**: Fixed seed=42

### Files
- `MNIST_Classifier_v1.ipynb` - Full dropout implementation
- `MNIST_Classifier_v2.ipynb` - Reduced dropout implementation  
- `MNIST_Classifier_Final.ipynb` - Batch normalization implementation
- `TrainingHelper.py` - Training utilities and device management
- `Visualization.py` - Plotting and metrics visualization