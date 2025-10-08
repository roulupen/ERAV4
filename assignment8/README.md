# Assignment 8: CIFAR-100 Classification with ResNet Architecture model. 

## 🎯 Project Goal

Train a ResNet architecture model on the CIFAR-100 dataset to achieve a target accuracy of 73% and deploy the model to Hugging Face for inference.

**Deployed Model:** [Hugging Face App](https://huggingface.co/spaces/roulupen/image-classification)

## 📊 Dataset Overview

**CIFAR-100:**
- 100 classes (e.g., apple, aquarium_fish, baby, bear, etc.)
- 60,000 32×32 color images
- 50,000 training images (500 per class)
- 10,000 test images (100 per class)
- Mean: [0.5071, 0.4867, 0.4408]
- Std: [0.2675, 0.2565, 0.2761]

## 🏗️ Model Architecture

### ResNet34 Architecture Details

```
Input: (3, 32, 32)
├── Initial Conv: 3x3, 64 channels (32x32)
├── Layer 1: 3x BasicBlock, 64 channels (32x32)
├── Layer 2: 4x BasicBlock, 128 channels (16x16)
├── Layer 3: 6x BasicBlock, 256 channels (8x8)
├── Layer 4: 3x BasicBlock, 512 channels (4x4)
├── Global Average Pooling: (4x4) → (1x1)
├── Dropout: p=0.1
└── Fully Connected: 512 → 100 classes
```

### Model Parameters
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
         Dropout2d-4           [-1, 64, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
              ReLU-7           [-1, 64, 32, 32]               0
         Dropout2d-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 64, 32, 32]          36,864
      BatchNorm2d-10           [-1, 64, 32, 32]             128
        Dropout2d-11           [-1, 64, 32, 32]               0
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13           [-1, 64, 32, 32]          36,864
      BatchNorm2d-14           [-1, 64, 32, 32]             128
             ReLU-15           [-1, 64, 32, 32]               0
        Dropout2d-16           [-1, 64, 32, 32]               0
           Conv2d-17           [-1, 64, 32, 32]          36,864
      BatchNorm2d-18           [-1, 64, 32, 32]             128
        Dropout2d-19           [-1, 64, 32, 32]               0
       BasicBlock-20           [-1, 64, 32, 32]               0
           Conv2d-21           [-1, 64, 32, 32]          36,864
      BatchNorm2d-22           [-1, 64, 32, 32]             128
             ReLU-23           [-1, 64, 32, 32]               0
        Dropout2d-24           [-1, 64, 32, 32]               0
           Conv2d-25           [-1, 64, 32, 32]          36,864
      BatchNorm2d-26           [-1, 64, 32, 32]             128
        Dropout2d-27           [-1, 64, 32, 32]               0
       BasicBlock-28           [-1, 64, 32, 32]               0
           Conv2d-29          [-1, 128, 16, 16]          73,728
      BatchNorm2d-30          [-1, 128, 16, 16]             256
             ReLU-31          [-1, 128, 16, 16]               0
        Dropout2d-32          [-1, 128, 16, 16]               0
           Conv2d-33          [-1, 128, 16, 16]         147,456
      BatchNorm2d-34          [-1, 128, 16, 16]             256
        Dropout2d-35          [-1, 128, 16, 16]               0
           Conv2d-36          [-1, 128, 16, 16]           8,192
           Conv2d-37          [-1, 128, 16, 16]           8,192
      BatchNorm2d-38          [-1, 128, 16, 16]             256
      BatchNorm2d-39          [-1, 128, 16, 16]             256
        Dropout2d-40          [-1, 128, 16, 16]               0
        Dropout2d-41          [-1, 128, 16, 16]               0
       BasicBlock-42          [-1, 128, 16, 16]               0
           Conv2d-43          [-1, 128, 16, 16]         147,456
      BatchNorm2d-44          [-1, 128, 16, 16]             256
             ReLU-45          [-1, 128, 16, 16]               0
        Dropout2d-46          [-1, 128, 16, 16]               0
           Conv2d-47          [-1, 128, 16, 16]         147,456
      BatchNorm2d-48          [-1, 128, 16, 16]             256
        Dropout2d-49          [-1, 128, 16, 16]               0
       BasicBlock-50          [-1, 128, 16, 16]               0
           Conv2d-51          [-1, 128, 16, 16]         147,456
      BatchNorm2d-52          [-1, 128, 16, 16]             256
             ReLU-53          [-1, 128, 16, 16]               0
        Dropout2d-54          [-1, 128, 16, 16]               0
           Conv2d-55          [-1, 128, 16, 16]         147,456
      BatchNorm2d-56          [-1, 128, 16, 16]             256
        Dropout2d-57          [-1, 128, 16, 16]               0
       BasicBlock-58          [-1, 128, 16, 16]               0
           Conv2d-59          [-1, 128, 16, 16]         147,456
      BatchNorm2d-60          [-1, 128, 16, 16]             256
             ReLU-61          [-1, 128, 16, 16]               0
        Dropout2d-62          [-1, 128, 16, 16]               0
           Conv2d-63          [-1, 128, 16, 16]         147,456
      BatchNorm2d-64          [-1, 128, 16, 16]             256
        Dropout2d-65          [-1, 128, 16, 16]               0
       BasicBlock-66          [-1, 128, 16, 16]               0
           Conv2d-67            [-1, 256, 8, 8]         294,912
      BatchNorm2d-68            [-1, 256, 8, 8]             512
             ReLU-69            [-1, 256, 8, 8]               0
        Dropout2d-70            [-1, 256, 8, 8]               0
           Conv2d-71            [-1, 256, 8, 8]         589,824
      BatchNorm2d-72            [-1, 256, 8, 8]             512
        Dropout2d-73            [-1, 256, 8, 8]               0
           Conv2d-74            [-1, 256, 8, 8]          32,768
           Conv2d-75            [-1, 256, 8, 8]          32,768
      BatchNorm2d-76            [-1, 256, 8, 8]             512
      BatchNorm2d-77            [-1, 256, 8, 8]             512
        Dropout2d-78            [-1, 256, 8, 8]               0
        Dropout2d-79            [-1, 256, 8, 8]               0
       BasicBlock-80            [-1, 256, 8, 8]               0
           Conv2d-81            [-1, 256, 8, 8]         589,824
      BatchNorm2d-82            [-1, 256, 8, 8]             512
             ReLU-83            [-1, 256, 8, 8]               0
        Dropout2d-84            [-1, 256, 8, 8]               0
           Conv2d-85            [-1, 256, 8, 8]         589,824
      BatchNorm2d-86            [-1, 256, 8, 8]             512
        Dropout2d-87            [-1, 256, 8, 8]               0
       BasicBlock-88            [-1, 256, 8, 8]               0
           Conv2d-89            [-1, 256, 8, 8]         589,824
      BatchNorm2d-90            [-1, 256, 8, 8]             512
             ReLU-91            [-1, 256, 8, 8]               0
        Dropout2d-92            [-1, 256, 8, 8]               0
           Conv2d-93            [-1, 256, 8, 8]         589,824
      BatchNorm2d-94            [-1, 256, 8, 8]             512
        Dropout2d-95            [-1, 256, 8, 8]               0
       BasicBlock-96            [-1, 256, 8, 8]               0
           Conv2d-97            [-1, 256, 8, 8]         589,824
      BatchNorm2d-98            [-1, 256, 8, 8]             512
             ReLU-99            [-1, 256, 8, 8]               0
       Dropout2d-100            [-1, 256, 8, 8]               0
          Conv2d-101            [-1, 256, 8, 8]         589,824
     BatchNorm2d-102            [-1, 256, 8, 8]             512
       Dropout2d-103            [-1, 256, 8, 8]               0
      BasicBlock-104            [-1, 256, 8, 8]               0
          Conv2d-105            [-1, 256, 8, 8]         589,824
     BatchNorm2d-106            [-1, 256, 8, 8]             512
            ReLU-107            [-1, 256, 8, 8]               0
       Dropout2d-108            [-1, 256, 8, 8]               0
          Conv2d-109            [-1, 256, 8, 8]         589,824
     BatchNorm2d-110            [-1, 256, 8, 8]             512
       Dropout2d-111            [-1, 256, 8, 8]               0
      BasicBlock-112            [-1, 256, 8, 8]               0
          Conv2d-113            [-1, 256, 8, 8]         589,824
     BatchNorm2d-114            [-1, 256, 8, 8]             512
            ReLU-115            [-1, 256, 8, 8]               0
       Dropout2d-116            [-1, 256, 8, 8]               0
          Conv2d-117            [-1, 256, 8, 8]         589,824
     BatchNorm2d-118            [-1, 256, 8, 8]             512
       Dropout2d-119            [-1, 256, 8, 8]               0
      BasicBlock-120            [-1, 256, 8, 8]               0
          Conv2d-121            [-1, 512, 4, 4]       1,179,648
     BatchNorm2d-122            [-1, 512, 4, 4]           1,024
            ReLU-123            [-1, 512, 4, 4]               0
       Dropout2d-124            [-1, 512, 4, 4]               0
          Conv2d-125            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-126            [-1, 512, 4, 4]           1,024
       Dropout2d-127            [-1, 512, 4, 4]               0
          Conv2d-128            [-1, 512, 4, 4]         131,072
          Conv2d-129            [-1, 512, 4, 4]         131,072
     BatchNorm2d-130            [-1, 512, 4, 4]           1,024
     BatchNorm2d-131            [-1, 512, 4, 4]           1,024
       Dropout2d-132            [-1, 512, 4, 4]               0
       Dropout2d-133            [-1, 512, 4, 4]               0
      BasicBlock-134            [-1, 512, 4, 4]               0
          Conv2d-135            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-136            [-1, 512, 4, 4]           1,024
            ReLU-137            [-1, 512, 4, 4]               0
       Dropout2d-138            [-1, 512, 4, 4]               0
          Conv2d-139            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-140            [-1, 512, 4, 4]           1,024
       Dropout2d-141            [-1, 512, 4, 4]               0
      BasicBlock-142            [-1, 512, 4, 4]               0
          Conv2d-143            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-144            [-1, 512, 4, 4]           1,024
            ReLU-145            [-1, 512, 4, 4]               0
       Dropout2d-146            [-1, 512, 4, 4]               0
          Conv2d-147            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-148            [-1, 512, 4, 4]           1,024
       Dropout2d-149            [-1, 512, 4, 4]               0
      BasicBlock-150            [-1, 512, 4, 4]               0
AdaptiveAvgPool2d-151            [-1, 512, 1, 1]               0
          Linear-152                  [-1, 100]          51,300
================================================================
Total params: 21,502,116
Trainable params: 21,502,116
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 32.13
Params size (MB): 82.02
Estimated Total Size (MB): 114.17
----------------------------------------------------------------
```
- **Total Parameters:** 21,502,116
- **Trainable Parameters:** 21,502,116
- **Architecture:** ResNet34 (deeper than ResNet18)

## 🔄 Data Augmentation Techniques

We implemented comprehensive data augmentation using Albumentations library to improve model generalization:

### Augmentation Pipeline:
1. **Pad + Random Crop** - Padding to 40x40, then random crop to 32x32
2. **Horizontal Flip** - 50% probability for horizontal flipping
3. **Affine Transformations** - Rotation, translation, and scaling
4. **Brightness/Contrast Adjustments** - Random brightness and contrast changes
5. **HSV Color Jitter** - Hue, saturation, and value adjustments
6. **CoarseDropout (Cutout)** - Random rectangular patches set to zero

## 📈 Learning Rate Scheduler and Optimizer

### Optimizer: AdamW
- **Learning Rate:** 0.001 (initial)
- **Weight Decay:** 0.0 (L2 regularization disabled)
- **Benefits:** Better generalization than Adam, adaptive learning rates

### Scheduler: OneCycleLR
- **Max Learning Rate:** 0.01
- **Strategy:** Cosine annealing
- **Ramp-up Period:** 30% of training (30 epochs)


## 📊 Training Logs

### Training Configuration:
- **Epochs:** 100 (completed full training)
- **Batch Size:** 128
- **Device:** CUDA GPU (Tesla T4)
- **Training Time:** ~130 minutes (2.2 hours)

### Training Progress:

```

🚀 Training for 100 epochs...
📈 Using OneCycleLR scheduler (updates per batch)
📁 Directory structure created in: ./checkpoints
🚀 Starting training for 100 epochs...
⏰ Training started at: 2025-10-08 10:49:11
📁 Checkpoint directory: ./checkpoints
Epoch  1/100 | Train: 4.2351 (5.34%) | Test: 3.8802 (9.88%) | LR: 0.000426 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 9.88%)
Epoch  2/100 | Train: 3.8473 (10.41%) | Test: 3.4504 (16.61%) | LR: 0.000505 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 16.61%)
Epoch  3/100 | Train: 3.5644 (15.14%) | Test: 3.1787 (22.02%) | LR: 0.000635 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 22.02%)
Epoch  4/100 | Train: 3.3245 (19.29%) | Test: 2.8681 (27.29%) | LR: 0.000815 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 27.29%)
Epoch  5/100 | Train: 3.1012 (23.40%) | Test: 2.7222 (32.17%) | LR: 0.001043 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 32.17%)
Epoch  6/100 | Train: 2.8934 (27.17%) | Test: 2.4768 (35.74%) | LR: 0.001317 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 35.74%)
Epoch  7/100 | Train: 2.7128 (30.69%) | Test: 2.2927 (39.27%) | LR: 0.001633 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 39.27%)
Epoch  8/100 | Train: 2.5599 (34.14%) | Test: 2.1474 (42.45%) | LR: 0.001988 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 42.45%)
Epoch  9/100 | Train: 2.4410 (36.46%) | Test: 1.9993 (45.69%) | LR: 0.002379 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 45.69%)
Epoch 10/100 | Train: 2.3332 (38.83%) | Test: 2.1006 (44.16%) | LR: 0.002800 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 11/100 | Train: 2.2386 (40.75%) | Test: 1.8324 (49.81%) | LR: 0.003248 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 49.81%)
Epoch 12/100 | Train: 2.1519 (42.57%) | Test: 1.7924 (50.90%) | LR: 0.003717 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 50.90%)
Epoch 13/100 | Train: 2.0868 (44.34%) | Test: 1.7446 (52.43%) | LR: 0.004203 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 52.43%)
Epoch 14/100 | Train: 2.0221 (45.81%) | Test: 1.6833 (53.44%) | LR: 0.004699 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 53.44%)
Epoch 15/100 | Train: 1.9595 (47.16%) | Test: 1.6391 (54.80%) | LR: 0.005201 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 54.80%)
Epoch 16/100 | Train: 1.9181 (48.22%) | Test: 1.6538 (54.33%) | LR: 0.005702 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 17/100 | Train: 1.8569 (49.72%) | Test: 1.5296 (57.35%) | LR: 0.006199 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 57.35%)
Epoch 18/100 | Train: 1.8163 (50.43%) | Test: 1.4810 (58.86%) | LR: 0.006684 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 58.86%)
Epoch 19/100 | Train: 1.7724 (51.69%) | Test: 1.4809 (58.62%) | LR: 0.007153 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 20/100 | Train: 1.7587 (51.78%) | Test: 1.4491 (59.55%) | LR: 0.007601 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 59.55%)
Epoch 21/100 | Train: 1.7010 (53.32%) | Test: 1.4605 (59.34%) | LR: 0.008022 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 22/100 | Train: 1.6748 (53.77%) | Test: 1.3858 (61.35%) | LR: 0.008413 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 61.35%)
Epoch 23/100 | Train: 1.6437 (54.62%) | Test: 1.4140 (61.20%) | LR: 0.008768 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 24/100 | Train: 1.6128 (55.38%) | Test: 1.3058 (63.33%) | LR: 0.009084 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 63.33%)
Epoch 25/100 | Train: 1.5708 (56.34%) | Test: 1.3644 (61.89%) | LR: 0.009357 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 26/100 | Train: 1.5590 (56.81%) | Test: 1.3166 (62.86%) | LR: 0.009585 | Time: 1.3m
  ⏳ No improvement (2/15)
Epoch 27/100 | Train: 1.5220 (57.63%) | Test: 1.3341 (63.19%) | LR: 0.009765 | Time: 1.3m
  ⏳ No improvement (3/15)
Epoch 28/100 | Train: 1.4882 (58.40%) | Test: 1.3010 (63.61%) | LR: 0.009895 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 63.61%)
Epoch 29/100 | Train: 1.4709 (58.76%) | Test: 1.2887 (64.19%) | LR: 0.009974 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 64.19%)
Epoch 30/100 | Train: 1.4402 (59.44%) | Test: 1.2567 (65.30%) | LR: 0.010000 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 65.30%)
Epoch 31/100 | Train: 1.4057 (60.40%) | Test: 1.2439 (65.01%) | LR: 0.009995 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 32/100 | Train: 1.3627 (61.60%) | Test: 1.2559 (65.64%) | LR: 0.009980 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 65.64%)
Epoch 33/100 | Train: 1.3349 (62.25%) | Test: 1.2479 (65.97%) | LR: 0.009955 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 65.97%)
Epoch 34/100 | Train: 1.3098 (62.80%) | Test: 1.2619 (66.07%) | LR: 0.009920 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 66.07%)
Epoch 35/100 | Train: 1.2850 (63.35%) | Test: 1.2268 (66.47%) | LR: 0.009875 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 66.47%)
Epoch 36/100 | Train: 1.2501 (64.23%) | Test: 1.1969 (67.54%) | LR: 0.009820 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 67.54%)
Epoch 37/100 | Train: 1.2311 (64.84%) | Test: 1.1790 (67.57%) | LR: 0.009755 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 67.57%)
Epoch 38/100 | Train: 1.2075 (65.33%) | Test: 1.1847 (67.52%) | LR: 0.009681 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 39/100 | Train: 1.1758 (66.05%) | Test: 1.1754 (68.24%) | LR: 0.009597 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 68.24%)
Epoch 40/100 | Train: 1.1553 (66.69%) | Test: 1.1856 (68.26%) | LR: 0.009505 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 68.26%)
Epoch 41/100 | Train: 1.1314 (67.29%) | Test: 1.1841 (68.16%) | LR: 0.009403 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 42/100 | Train: 1.0993 (68.26%) | Test: 1.1595 (69.09%) | LR: 0.009292 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 69.09%)
Epoch 43/100 | Train: 1.0814 (68.52%) | Test: 1.1560 (69.39%) | LR: 0.009173 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 69.39%)
Epoch 44/100 | Train: 1.0584 (69.23%) | Test: 1.1607 (69.51%) | LR: 0.009045 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 69.51%)
Epoch 45/100 | Train: 1.0258 (70.27%) | Test: 1.1455 (69.32%) | LR: 0.008909 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 46/100 | Train: 1.0145 (70.41%) | Test: 1.1443 (69.68%) | LR: 0.008765 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 69.68%)
Epoch 47/100 | Train: 0.9870 (71.35%) | Test: 1.1360 (70.57%) | LR: 0.008614 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 70.57%)
Epoch 48/100 | Train: 0.9642 (71.75%) | Test: 1.1237 (70.33%) | LR: 0.008455 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 49/100 | Train: 0.9417 (72.38%) | Test: 1.1594 (70.15%) | LR: 0.008289 | Time: 1.3m
  ⏳ No improvement (2/15)
Epoch 50/100 | Train: 0.9144 (73.18%) | Test: 1.1525 (70.54%) | LR: 0.008117 | Time: 1.3m
  ⏳ No improvement (3/15)
Epoch 51/100 | Train: 0.8902 (73.79%) | Test: 1.1557 (70.77%) | LR: 0.007938 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 70.77%)
Epoch 52/100 | Train: 0.8805 (74.13%) | Test: 1.1602 (70.89%) | LR: 0.007754 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 70.89%)
Epoch 53/100 | Train: 0.8503 (74.75%) | Test: 1.1404 (71.54%) | LR: 0.007564 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 71.54%)
Epoch 54/100 | Train: 0.8283 (75.50%) | Test: 1.1613 (71.86%) | LR: 0.007369 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 71.86%)
Epoch 55/100 | Train: 0.8093 (76.05%) | Test: 1.1613 (71.94%) | LR: 0.007169 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 71.94%)
Epoch 56/100 | Train: 0.7835 (76.65%) | Test: 1.1528 (71.66%) | LR: 0.006965 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 57/100 | Train: 0.7669 (77.20%) | Test: 1.1422 (72.01%) | LR: 0.006756 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 72.01%)
Epoch 58/100 | Train: 0.7450 (77.79%) | Test: 1.1591 (71.84%) | LR: 0.006545 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 59/100 | Train: 0.7256 (78.41%) | Test: 1.1281 (72.56%) | LR: 0.006330 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 72.56%)
Epoch 60/100 | Train: 0.7094 (78.73%) | Test: 1.1354 (72.15%) | LR: 0.006112 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 61/100 | Train: 0.6856 (79.52%) | Test: 1.1237 (72.63%) | LR: 0.005892 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 72.63%)
Epoch 62/100 | Train: 0.6697 (79.87%) | Test: 1.1701 (72.16%) | LR: 0.005671 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 63/100 | Train: 0.6473 (80.34%) | Test: 1.1376 (72.83%) | LR: 0.005448 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 72.83%)
Epoch 64/100 | Train: 0.6272 (81.08%) | Test: 1.1341 (73.05%) | LR: 0.005224 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 73.05%)
Epoch 65/100 | Train: 0.6008 (82.06%) | Test: 1.1441 (73.18%) | LR: 0.004999 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 73.18%)
Epoch 66/100 | Train: 0.5954 (82.08%) | Test: 1.1368 (73.32%) | LR: 0.004775 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 73.32%)
Epoch 67/100 | Train: 0.5857 (82.07%) | Test: 1.1705 (72.81%) | LR: 0.004551 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 68/100 | Train: 0.5603 (83.23%) | Test: 1.1555 (73.04%) | LR: 0.004328 | Time: 1.3m
  ⏳ No improvement (2/15)
Epoch 69/100 | Train: 0.5460 (83.45%) | Test: 1.1739 (73.29%) | LR: 0.004107 | Time: 1.3m
  ⏳ No improvement (3/15)
Epoch 70/100 | Train: 0.5237 (84.18%) | Test: 1.1396 (74.02%) | LR: 0.003887 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.02%)
Epoch 71/100 | Train: 0.5108 (84.65%) | Test: 1.1572 (73.99%) | LR: 0.003669 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 72/100 | Train: 0.5030 (84.66%) | Test: 1.1888 (73.53%) | LR: 0.003454 | Time: 1.3m
  ⏳ No improvement (2/15)
Epoch 73/100 | Train: 0.4868 (85.21%) | Test: 1.1684 (73.87%) | LR: 0.003243 | Time: 1.3m
  ⏳ No improvement (3/15)
Epoch 74/100 | Train: 0.4669 (85.62%) | Test: 1.1772 (73.68%) | LR: 0.003034 | Time: 1.3m
  ⏳ No improvement (4/15)
Epoch 75/100 | Train: 0.4602 (86.16%) | Test: 1.1660 (74.18%) | LR: 0.002830 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.18%)
Epoch 76/100 | Train: 0.4396 (86.54%) | Test: 1.1656 (73.84%) | LR: 0.002630 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 77/100 | Train: 0.4300 (86.90%) | Test: 1.1685 (74.27%) | LR: 0.002435 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.27%)
Epoch 78/100 | Train: 0.4200 (87.19%) | Test: 1.2050 (74.39%) | LR: 0.002245 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.39%)
Epoch 79/100 | Train: 0.4049 (87.64%) | Test: 1.1895 (74.58%) | LR: 0.002061 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.58%)
Epoch 80/100 | Train: 0.3970 (87.83%) | Test: 1.2000 (74.37%) | LR: 0.001882 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 81/100 | Train: 0.3908 (88.09%) | Test: 1.1745 (74.84%) | LR: 0.001710 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.84%)
Epoch 82/100 | Train: 0.3821 (88.41%) | Test: 1.1753 (74.63%) | LR: 0.001544 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 83/100 | Train: 0.3664 (88.86%) | Test: 1.1989 (74.94%) | LR: 0.001386 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.94%)
Epoch 84/100 | Train: 0.3537 (89.33%) | Test: 1.1905 (74.95%) | LR: 0.001234 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.95%)
Epoch 85/100 | Train: 0.3513 (89.30%) | Test: 1.1929 (74.95%) | LR: 0.001091 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 86/100 | Train: 0.3515 (89.14%) | Test: 1.1921 (74.97%) | LR: 0.000955 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 74.97%)
Epoch 87/100 | Train: 0.3381 (89.67%) | Test: 1.2067 (74.89%) | LR: 0.000827 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 88/100 | Train: 0.3308 (89.84%) | Test: 1.1884 (74.70%) | LR: 0.000707 | Time: 1.3m
  ⏳ No improvement (2/15)
Epoch 89/100 | Train: 0.3216 (90.28%) | Test: 1.1944 (74.97%) | LR: 0.000597 | Time: 1.3m
  ⏳ No improvement (3/15)
Epoch 90/100 | Train: 0.3155 (90.31%) | Test: 1.2192 (75.19%) | LR: 0.000495 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 75.19%)
Epoch 91/100 | Train: 0.3212 (90.26%) | Test: 1.2026 (74.90%) | LR: 0.000402 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 92/100 | Train: 0.3182 (90.22%) | Test: 1.1934 (75.18%) | LR: 0.000319 | Time: 1.3m
  ⏳ No improvement (2/15)
Epoch 93/100 | Train: 0.3158 (90.25%) | Test: 1.1941 (74.98%) | LR: 0.000245 | Time: 1.3m
  ⏳ No improvement (3/15)
Epoch 94/100 | Train: 0.3097 (90.52%) | Test: 1.2024 (75.10%) | LR: 0.000180 | Time: 1.3m
  ⏳ No improvement (4/15)
Epoch 95/100 | Train: 0.3149 (90.41%) | Test: 1.1905 (75.25%) | LR: 0.000125 | Time: 1.3m
  🏆 New best model saved! (Test Acc: 75.25%)
Epoch 96/100 | Train: 0.3053 (90.58%) | Test: 1.1891 (75.16%) | LR: 0.000080 | Time: 1.3m
  ⏳ No improvement (1/15)
Epoch 97/100 | Train: 0.3072 (90.56%) | Test: 1.1924 (75.14%) | LR: 0.000045 | Time: 1.3m
  ⏳ No improvement (2/15)
Epoch 98/100 | Train: 0.3099 (90.64%) | Test: 1.2006 (75.13%) | LR: 0.000020 | Time: 1.3m
  ⏳ No improvement (3/15)
Epoch 99/100 | Train: 0.3016 (90.74%) | Test: 1.1995 (75.24%) | LR: 0.000005 | Time: 1.3m
  ⏳ No improvement (4/15)
Epoch 100/100 | Train: 0.3045 (90.62%) | Test: 1.1953 (75.05%) | LR: 0.000000 | Time: 1.3m
  ⏳ No improvement (5/15)

✅ Training completed!
   Total time: 2.2h
   Best test accuracy: 75.25% (Epoch 95)

⏰ Training time: 132.0 minutes

```

### Final Results:
- **Best Test Accuracy:** 75.25%
- **Final Test Accuracy:** 75.05%
- **Best Epoch:** 95
- **Overfitting Gap:** 15.57%
- **Target Achievement:** ✅ (Target: 73%, Achieved: 75.25%)


## 🚀 Inference and Deployment

### Hugging Face Deployment
Our trained model has been deployed to Hugging Face Spaces for easy inference:

**🔗 Live Demo:** [https://huggingface.co/spaces/roulupen/image-classification](https://huggingface.co/spaces/roulupen/image-classification)

### Features:
- **Interactive Interface:** Upload images and get instant predictions
- **Top-5 Predictions:** Shows confidence scores for top 5 classes
- **Real-time Inference:** Fast prediction on uploaded images
- **CIFAR-100 Classes:** Full 100-class classification support

### Sample Results

#### Test Image 1: boy-1.png
![boy-1](./boy-1.png?raw=true)
```
Predicted Class: boy
Confidence: 82.7%
Top 5 Predictions:
1. boy (82.7%)
2. man (10.2%)
3. baby (4.1%)
4. woman (2.3%)
5. girl (0.7%)
```

#### Test Image 2: girl-1.png
![girl-1](./girl-1.png?raw=true)
```
Predicted Class: girl
Confidence: 81.4%
Top 5 Predictions:
1. girl (81.4%)
2. woman (11.8%)
3. baby (4.2%)
4. boy (1.8%)
5. man (0.8%)
```

### Using the Deployed Model
1. Visit: [https://huggingface.co/spaces/roulupen/image-classification](https://huggingface.co/spaces/roulupen/image-classification)
2. Upload an image (32x32 recommended for best results)
3. Get instant predictions with confidence scores


## 🎉 Conclusion

### Goals Achieved:
1. ✅ **Target Accuracy:** Achieved 75.25% (target: 73%) - **Exceeded by 2.25%**
2. ✅ **Model Deployment:** Successfully deployed to Hugging Face
3. ✅ **Architecture Implementation:** Complete ResNet34 from scratch
4. ✅ **Data Augmentation:** Comprehensive augmentation pipeline
5. ✅ **Training Optimization:** OneCycleLR scheduler for efficient training

**🔗 Live Demo:** [https://huggingface.co/spaces/roulupen/image-classification](https://huggingface.co/spaces/roulupen/image-classification)
