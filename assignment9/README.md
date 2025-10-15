# Assignment 9: ResNet50 ImageNet Training

This assignment implements ResNet50 training on ImageNet-1K dataset with 1000 classes.

## 📁 Project Structure

```
assignment9/
├── model.py              # ResNet50 architecture for ImageNet
├── data.py               # ImageNet data loading and augmentation
├── config.py             # Training configuration management
├── main.py               # Main training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── ResNet50_ImageNet_Training.ipynb  # Kaggle notebook for testing
```

## 🏗️ Model Architecture

**ResNet50** - Deep residual network with 50 layers:
- Input: 224×224×3 (ImageNet images)
- Initial Conv: 7×7 conv with stride=2, 64 channels
- Max Pooling: 3×3 with stride=2
- Layer 1: 3 Bottleneck blocks (64 channels, 56×56)
- Layer 2: 4 Bottleneck blocks (128 channels, 28×28)
- Layer 3: 6 Bottleneck blocks (256 channels, 14×14)
- Layer 4: 3 Bottleneck blocks (512 channels, 7×7)
- Global Average Pooling: 7×7 → 1×1
- Fully Connected: 2048 → 1000 classes
- **Total Parameters: ~25.5M**

### Key Features:
- ✅ Bottleneck blocks with 1×1 → 3×3 → 1×1 convolutions
- ✅ Residual connections (skip connections)
- ✅ Batch Normalization in every block
- ✅ Optional dropout for regularization
- ✅ Kaiming weight initialization
- ✅ Support for pretrained weights

## 📊 Dataset

**ImageNet-1K (ILSVRC2012)**:
- Training: 1,281,167 images
- Validation: 50,000 images
- Classes: 1000 object categories
- Image size: Variable (resized to 224×224)

### Data Augmentation:
- **Training**:
  - RandomResizedCrop(224)
  - HorizontalFlip
  - ColorJitter
  - GaussianBlur/Noise
  - Normalization (ImageNet stats)

- **Validation**:
  - Resize(256)
  - CenterCrop(224)
  - Normalization

## 🚀 Usage

### Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train ResNet50 on ImageNet
python main.py --epochs 90 --batch_size 128

# Test architecture only (no training)
python main.py --test-architecture
```

### Kaggle Training

1. Open `ResNet50_ImageNet_Training.ipynb` in Kaggle
2. Add ImageNet dataset to your notebook:
   - Dataset: `imagenet-object-localization-challenge`
3. Enable GPU accelerator
4. Run all cells

The notebook is configured to:
- ✅ Automatically detect Kaggle environment
- ✅ Use correct ImageNet paths
- ✅ Install required packages
- ✅ Train for 1 batch with 128 batch size (for testing)
- ✅ Save model checkpoints

## ⚙️ Configuration

Edit `config.py` to modify training parameters:

```python
# Model
model_name: 'resnet50_imagenet'
num_classes: 1000
dropout: 0.0

# Data
batch_size: 128
num_workers: 4
data_dir: '/kaggle/input/imagenet-object-localization-challenge'

# Training
epochs: 90
learning_rate: 0.1
weight_decay: 1e-4
optimizer: 'sgd'
scheduler: 'step'

# Target
target_accuracy: 70.0  # Top-1 accuracy
```

## 📈 Training Strategy

1. **Optimizer**: SGD with momentum (0.9)
2. **Learning Rate**: 0.1 with step decay (÷10 every 30 epochs)
3. **Weight Decay**: 1e-4 for regularization
4. **Batch Size**: 128 (can be adjusted based on GPU memory)
5. **Epochs**: 90 (standard ImageNet training)
6. **Early Stopping**: Patience of 10 epochs

## 🎯 Expected Results

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | ~70-76% |
| Top-5 Accuracy | ~90-93% |
| Parameters | 25.5M |
| Training Time | ~3-7 days (single GPU) |

## 📝 Notes

### For Kaggle Users:
- The notebook is designed to run on Kaggle's free GPU tier
- ImageNet dataset is available via Kaggle Datasets
- Training is limited to 1 batch for testing purposes
- Adjust `limit_samples` parameter to control dataset size

### Memory Considerations:
- Batch size 128 requires ~16GB GPU memory
- Reduce batch size if you encounter OOM errors
- Use mixed precision training (AMP) for faster training

### Directory Structure:
The code automatically detects Kaggle ImageNet structure:
```
/kaggle/input/imagenet-object-localization-challenge/
└── ILSVRC/
    └── Data/
        └── CLS-LOC/
            ├── train/
            │   ├── n01440764/
            │   ├── n01443537/
            │   └── ...
            └── val/
                ├── n01440764/
                ├── n01443537/
                └── ...
```

## 🔧 Code Reuse from Common Directory

This assignment reuses the following from `common/`:
- ✅ `trainer.py`: Generic training loop with early stopping
- ✅ `utils.py`: Device detection, checkpointing, plotting
- ✅ `config.py`: Configuration management (adapted for ImageNet)

## 📚 References

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575)
3. [PyTorch ResNet Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

## 🐛 Troubleshooting

**Issue**: OOM (Out of Memory) errors
- Solution: Reduce batch size or enable mixed precision training

**Issue**: Dataset not found on Kaggle
- Solution: Ensure you've added the ImageNet dataset to your notebook

**Issue**: Slow data loading
- Solution: Increase `num_workers` or use persistent workers

## 📄 License

This project is part of the ERA V4 course assignments.

