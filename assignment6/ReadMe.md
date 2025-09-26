# 🧠 MNIST Model Training Experiments

This repository contains experiments with **lightweight CNN architectures for MNIST classification**, built using **modular code**.
All models are defined in [`model.py`](./model.py) as `MNISTNet4BlockWithBatchNormMaxPoolConvFinal`, `MNISTNet4BlockWithBatchNormMaxPoolDropoutConvFinal` and `MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling`.

Our **training targets** were refined step by step to balance **accuracy, efficiency, and generalization**:

### 🎯 Final Target

* **Accuracy:** ≥ **99.4%**, consistently in the last few epochs (not one-time spikes).
* **Epochs:** ≤ **15**.
* **Parameters:** ≤ **8,000**.

---

# 🚀 Model Experiments

## **MNISTNet4BlockWithBatchNormMaxPoolConvFinal – CNN with Batch Normalization**

**Architecture (high-level):**
A small CNN with convolutional layers followed by **Batch Normalization** to stabilize training and improve convergence.

### 📄 Training Logs

**Command To Run the Model**: `python train.py --model batchnorm --epochs 15 --batch_size 64 --scheduler cyclic  --dropout_prob 0.015`

```
(era_v4) roul@Upendras-MacBook-Pro assignment6 % python train.py --model batchnorm --epochs 15 --batch_size 64 --scheduler cyclic                       
🎯 MNIST Classification Training Pipeline
============================================================
📋 Configuration:
  Model: batchnorm
  Batch Size: 64
  Epochs: 15
  Learning Rate: 0.001
  Optimizer: ADAMW
  Scheduler: cyclic
  Dropout: 0.15
  Weight Decay: 0.001
  Cutout: ❌
  Random Seed: 42
============================================================
🌱 Random seed set to 42
✅ Using Apple Silicon GPU (Metal Performance Shaders)
🖥️  Using device: mps

📥 Loading MNIST dataset...
  Batch size: 64
  Workers: 4
  Data directory: ./data
📊 Data loaded successfully:
  Training samples: 60,000
  Test samples: 10,000
  Batch size: 64
  Training batches: 938
  Test batches: 157

🏗️ Setting up batchnorm model...
  Total parameters: 8,884
  Under 8k limit: ❌

📊 Model Architecture:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
              ReLU-3            [-1, 8, 28, 28]               0
            Conv2d-4           [-1, 16, 28, 28]           1,168
       BatchNorm2d-5           [-1, 16, 28, 28]              32
              ReLU-6           [-1, 16, 28, 28]               0
            Conv2d-7            [-1, 8, 28, 28]             136
       BatchNorm2d-8            [-1, 8, 28, 28]              16
              ReLU-9            [-1, 8, 28, 28]               0
        MaxPool2d-10            [-1, 8, 14, 14]               0
           Conv2d-11           [-1, 12, 14, 14]             876
      BatchNorm2d-12           [-1, 12, 14, 14]              24
             ReLU-13           [-1, 12, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           1,744
      BatchNorm2d-15           [-1, 16, 14, 14]              32
             ReLU-16           [-1, 16, 14, 14]               0
           Conv2d-17            [-1, 6, 14, 14]             102
      BatchNorm2d-18            [-1, 6, 14, 14]              12
             ReLU-19            [-1, 6, 14, 14]               0
        MaxPool2d-20              [-1, 6, 7, 7]               0
           Conv2d-21             [-1, 12, 7, 7]             660
      BatchNorm2d-22             [-1, 12, 7, 7]              24
             ReLU-23             [-1, 12, 7, 7]               0
           Conv2d-24             [-1, 12, 7, 7]           1,308
      BatchNorm2d-25             [-1, 12, 7, 7]              24
             ReLU-26             [-1, 12, 7, 7]               0
           Conv2d-27              [-1, 6, 7, 7]              78
      BatchNorm2d-28              [-1, 6, 7, 7]              12
             ReLU-29              [-1, 6, 7, 7]               0
        MaxPool2d-30              [-1, 6, 3, 3]               0
           Conv2d-31             [-1, 10, 3, 3]             550
      BatchNorm2d-32             [-1, 10, 3, 3]              20
             ReLU-33             [-1, 10, 3, 3]               0
           Conv2d-34             [-1, 10, 3, 3]             910
      BatchNorm2d-35             [-1, 10, 3, 3]              20
             ReLU-36             [-1, 10, 3, 3]               0
           Conv2d-37             [-1, 10, 1, 1]             910
      BatchNorm2d-38             [-1, 10, 1, 1]              20
             ReLU-39             [-1, 10, 1, 1]               0
          Flatten-40                   [-1, 10]               0
           Linear-41                   [-1, 10]             110
================================================================
Total params: 8,884
Trainable params: 8,884
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.78
Params size (MB): 0.03
Estimated Total Size (MB): 0.82
----------------------------------------------------------------

⚙️  Setting up ADAMW optimizer and cyclic scheduler...
  Loss function: NLLLoss

🚀 Starting training for 15 epochs...
⏰ Training started at: 2025-09-26 22:26:04
📁 Training Setup Complete:
  Epochs: 15
  Early stopping patience: 8
  Checkpoint directory: ./checkpoints
  Best model path: ./checkpoints/best_mnist_model.pth
  Latest checkpoint: ./checkpoints/latest_checkpoint.pth

🚀 Starting training for 15 epochs...
⏰ Training started at: 2025-09-26 22:26:04
Epoch  1/15 | Train: 1.3425 (68.23%) | Test: 0.3740 (95.55%) | LR: 0.000708 | Time: 10.5s                                                                                                     
  🏆 New best model saved! (Test Acc: 95.55%)
Epoch  2/15 | Train: 0.1902 (96.48%) | Test: 0.0957 (97.50%) | LR: 0.001460 | Time: 9.9s                                                                                                      
  🏆 New best model saved! (Test Acc: 97.50%)
Epoch  3/15 | Train: 0.0813 (97.81%) | Test: 0.0536 (98.45%) | LR: 0.001487 | Time: 9.9s                                                                                                      
  🏆 New best model saved! (Test Acc: 98.45%)
Epoch  4/15 | Train: 0.0606 (98.26%) | Test: 0.0406 (98.85%) | LR: 0.001431 | Time: 9.9s                                                                                                      
  🏆 New best model saved! (Test Acc: 98.85%)
Epoch  5/15 | Train: 0.0485 (98.56%) | Test: 0.0410 (98.75%) | LR: 0.001334 | Time: 9.9s                                                                                                      
  ⏳ No improvement (1/8)
Epoch  6/15 | Train: 0.0425 (98.73%) | Test: 0.0377 (98.83%) | LR: 0.001202 | Time: 10.0s                                                                                                     
  ⏳ No improvement (2/8)
Epoch  7/15 | Train: 0.0393 (98.82%) | Test: 0.0292 (99.09%) | LR: 0.001042 | Time: 10.0s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.09%)
Epoch  8/15 | Train: 0.0294 (99.08%) | Test: 0.0294 (99.13%) | LR: 0.000865 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.13%)
Epoch  9/15 | Train: 0.0266 (99.18%) | Test: 0.0226 (99.31%) | LR: 0.000681 | Time: 10.0s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.31%)
Epoch 10/15 | Train: 0.0203 (99.39%) | Test: 0.0216 (99.39%) | LR: 0.000501 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.39%)
Epoch 11/15 | Train: 0.0151 (99.55%) | Test: 0.0198 (99.35%) | LR: 0.000336 | Time: 10.1s                                                                                                     
  ⏳ No improvement (1/8)
Epoch 12/15 | Train: 0.0125 (99.60%) | Test: 0.0194 (99.45%) | LR: 0.000196 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.45%)
Epoch 13/15 | Train: 0.0094 (99.73%) | Test: 0.0194 (99.43%) | LR: 0.000089 | Time: 10.1s                                                                                                     
  ⏳ No improvement (1/8)
Epoch 14/15 | Train: 0.0081 (99.78%) | Test: 0.0185 (99.41%) | LR: 0.000023 | Time: 10.3s                                                                                                     
  ⏳ No improvement (2/8)
Epoch 15/15 | Train: 0.0071 (99.81%) | Test: 0.0189 (99.44%) | LR: 0.000000 | Time: 10.1s                                                                                                     
  ⏳ No improvement (3/8)

✅ Training completed!
   Total time: 2.53 minutes
   Best test accuracy: 99.45% (Epoch 12)

🎉 Training completed successfully!
📊 Final Results:
  Best Test Accuracy: 99.45%
  Final Test Accuracy: 99.44%
  Target Achievement: ✅
  Total Epochs: 15
```

### 📊 Analysis

* The model contains **8.88K parameters**, slightly above the target of **8K parameters**.
* The model reached **99% test accuracy by Epoch 7**.
* It achieved its **best accuracy of 99.45% at Epoch 12**.
* Final accuracy after 15 epochs was **99.44%**, which successfully meets the target of **≥99.4%**.
* Batch Normalization layers helped stabilize training and contributed to fast convergence.
* The model architecture is efficient and compact, but minor refinements (like GAP in place of FC) can optimize parameter usage further.
* The training accuracy reached **99.8%**, while the test accuracy plateaued around **99.4%**. This gap indicates **mild overfitting**.
* Adding **regularization techniques** such as **dropout, cutout, or data augmentation** in future runs can reduce this gap.
* **Conclusion**: The current batchnorm-based CNN achieved **99.45% accuracy**, successfully meeting the target within 15 epochs. The model is compact and efficient but can be further optimized by reducing parameters with GAP and improving generalization with regularization strategies (e.g., dropout, cutout).
---

## **MNISTNet4BlockWithBatchNormMaxPoolDropoutConvFinal – CNN with Dropout**

**Architecture (high-level):**
Similar CNN, but with **Dropout layers** for regularization instead of BatchNorm.

### 📄 Training Logs
**Command To Run the Model**: `python train.py --model dropout --epochs 15 --batch_size 64 --scheduler cyclic  --dropout_prob 0.015`
```
(era_v4) roul@Upendras-MacBook-Pro assignment6 % python train.py --model dropout --epochs 15 --batch_size 64 --scheduler cyclic  --dropout_prob 0.015 --verbose
🎯 MNIST Classification Training Pipeline
============================================================
📋 Configuration:
  Model: dropout
  Batch Size: 64
  Epochs: 15
  Learning Rate: 0.001
  Optimizer: ADAMW
  Scheduler: cyclic
  Dropout: 0.015
  Weight Decay: 0.001
  Cutout: ❌
  Random Seed: 42
============================================================
🌱 Random seed set to 42
✅ Using Apple Silicon GPU (Metal Performance Shaders)
🖥️  Using device: mps

📥 Loading MNIST dataset...
  Batch size: 64
  Workers: 4
  Data directory: ./data
📊 Data loaded successfully:
  Training samples: 60,000
  Test samples: 10,000
  Batch size: 64
  Training batches: 938
  Test batches: 157

🏗️ Setting up dropout model...
  Total parameters: 8,884
  Under 8k limit: ❌

📊 Model Architecture:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
              ReLU-3            [-1, 8, 28, 28]               0
           Dropout-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 16, 28, 28]           1,168
       BatchNorm2d-6           [-1, 16, 28, 28]              32
              ReLU-7           [-1, 16, 28, 28]               0
           Dropout-8           [-1, 16, 28, 28]               0
            Conv2d-9            [-1, 8, 28, 28]             136
      BatchNorm2d-10            [-1, 8, 28, 28]              16
             ReLU-11            [-1, 8, 28, 28]               0
        MaxPool2d-12            [-1, 8, 14, 14]               0
           Conv2d-13           [-1, 12, 14, 14]             876
      BatchNorm2d-14           [-1, 12, 14, 14]              24
             ReLU-15           [-1, 12, 14, 14]               0
          Dropout-16           [-1, 12, 14, 14]               0
           Conv2d-17           [-1, 16, 14, 14]           1,744
      BatchNorm2d-18           [-1, 16, 14, 14]              32
             ReLU-19           [-1, 16, 14, 14]               0
          Dropout-20           [-1, 16, 14, 14]               0
           Conv2d-21            [-1, 6, 14, 14]             102
      BatchNorm2d-22            [-1, 6, 14, 14]              12
             ReLU-23            [-1, 6, 14, 14]               0
        MaxPool2d-24              [-1, 6, 7, 7]               0
           Conv2d-25             [-1, 12, 7, 7]             660
      BatchNorm2d-26             [-1, 12, 7, 7]              24
             ReLU-27             [-1, 12, 7, 7]               0
           Conv2d-28             [-1, 12, 7, 7]           1,308
      BatchNorm2d-29             [-1, 12, 7, 7]              24
             ReLU-30             [-1, 12, 7, 7]               0
           Conv2d-31              [-1, 6, 7, 7]              78
      BatchNorm2d-32              [-1, 6, 7, 7]              12
             ReLU-33              [-1, 6, 7, 7]               0
        MaxPool2d-34              [-1, 6, 3, 3]               0
           Conv2d-35             [-1, 10, 3, 3]             550
      BatchNorm2d-36             [-1, 10, 3, 3]              20
             ReLU-37             [-1, 10, 3, 3]               0
           Conv2d-38             [-1, 10, 3, 3]             910
      BatchNorm2d-39             [-1, 10, 3, 3]              20
             ReLU-40             [-1, 10, 3, 3]               0
           Conv2d-41             [-1, 10, 1, 1]             910
      BatchNorm2d-42             [-1, 10, 1, 1]              20
             ReLU-43             [-1, 10, 1, 1]               0
          Flatten-44                   [-1, 10]               0
           Linear-45                   [-1, 10]             110
================================================================
Total params: 8,884
Trainable params: 8,884
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.96
Params size (MB): 0.03
Estimated Total Size (MB): 1.00
----------------------------------------------------------------

⚙️  Setting up ADAMW optimizer and cyclic scheduler...
  Loss function: NLLLoss

🚀 Starting training for 15 epochs...
⏰ Training started at: 2025-09-26 22:13:23
📁 Training Setup Complete:
  Epochs: 15
  Early stopping patience: 8
  Checkpoint directory: ./checkpoints
  Best model path: ./checkpoints/best_mnist_model.pth
  Latest checkpoint: ./checkpoints/latest_checkpoint.pth

🚀 Starting training for 15 epochs...
⏰ Training started at: 2025-09-26 22:13:23
Epoch  1/15 | Train: 1.3621 (67.14%) | Test: 0.4451 (94.11%) | LR: 0.000708 | Time: 10.7s                                                                                                     
  🏆 New best model saved! (Test Acc: 94.11%)
Epoch  2/15 | Train: 0.1966 (96.39%) | Test: 0.0917 (97.58%) | LR: 0.001460 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 97.58%)
Epoch  3/15 | Train: 0.0865 (97.64%) | Test: 0.0557 (98.43%) | LR: 0.001487 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.43%)
Epoch  4/15 | Train: 0.0620 (98.18%) | Test: 0.0355 (98.95%) | LR: 0.001431 | Time: 10.3s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.95%)
Epoch  5/15 | Train: 0.0544 (98.44%) | Test: 0.0489 (98.49%) | LR: 0.001334 | Time: 10.3s                                                                                                     
  ⏳ No improvement (1/8)
Epoch  6/15 | Train: 0.0454 (98.62%) | Test: 0.0366 (98.91%) | LR: 0.001202 | Time: 10.4s                                                                                                     
  ⏳ No improvement (2/8)
Epoch  7/15 | Train: 0.0382 (98.87%) | Test: 0.0328 (98.95%) | LR: 0.001042 | Time: 10.8s                                                                                                     
  ⏳ No improvement (3/8)
Epoch  8/15 | Train: 0.0351 (98.92%) | Test: 0.0279 (99.01%) | LR: 0.000865 | Time: 10.7s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.01%)
Epoch  9/15 | Train: 0.0278 (99.19%) | Test: 0.0251 (99.23%) | LR: 0.000681 | Time: 10.5s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.23%)
Epoch 10/15 | Train: 0.0251 (99.22%) | Test: 0.0237 (99.28%) | LR: 0.000501 | Time: 10.5s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.28%)
Epoch 11/15 | Train: 0.0204 (99.39%) | Test: 0.0235 (99.26%) | LR: 0.000336 | Time: 10.4s                                                                                                     
  ⏳ No improvement (1/8)
Epoch 12/15 | Train: 0.0168 (99.50%) | Test: 0.0216 (99.37%) | LR: 0.000196 | Time: 10.5s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.37%)
Epoch 13/15 | Train: 0.0145 (99.57%) | Test: 0.0224 (99.26%) | LR: 0.000089 | Time: 10.5s                                                                                                     
  ⏳ No improvement (1/8)
Epoch 14/15 | Train: 0.0124 (99.65%) | Test: 0.0213 (99.32%) | LR: 0.000023 | Time: 10.8s                                                                                                     
  ⏳ No improvement (2/8)
Epoch 15/15 | Train: 0.0119 (99.67%) | Test: 0.0213 (99.34%) | LR: 0.000000 | Time: 10.8s                                                                                                     
  ⏳ No improvement (3/8)

✅ Training completed!
   Total time: 2.64 minutes
   Best test accuracy: 99.37% (Epoch 12)

🎉 Training completed successfully!
📊 Final Results:
  Best Test Accuracy: 99.37%
  Final Test Accuracy: 99.34%
  Target Achievement: ❌
  Total Epochs: 15
```

### 📊 Analysis

* The model has **8.88K parameters**, same as the batchnorm version, which is still slightly above the **8K target**.
* Dropout was applied in the **earlier and mid-level convolutional layers**, but **not in the final layers**.
* This design choice helps preserve the critical feature representations learned in the deeper layers while still providing regularization in earlier parts of the network.
* The model crossed **99% test accuracy by Epoch 8**. It achieved its **best test accuracy of 99.37% at Epoch 12**. Final accuracy after 15 epochs was **99.34%**, which is **slightly below the 99.4% target**.
* **Conclusion:** The dropout-augmented CNN achieved **99.37% accuracy**, which is strong but just shy of the **99.4% target**. Dropout provided mild regularization, reducing overfitting compared to the batchnorm-only version, but further tuning (e.g., increasing dropout rate, adding cutout, or parameter reduction via GAP) is needed to both meet the accuracy target and reduce parameters below 8K.


---

## **MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling – GAP CNN**

**Architecture (high-level):**

* Uses **Global Average Pooling (GAP)** instead of a dense FC layer at the end.
* Dramatically reduces parameter count to **7,954 (<8K)**.
* No augmentations used — tested pure architecture efficiency.

### 📄 Training Logs
**Command To Run the Model**: `python train.py --model gap --epochs 15 --batch_size 64 --scheduler cyclic  --dropout_prob 0.015`
```
🚀 Starting training for 15 epochs...
⏰ Training started at: 2025-09-26 17:18:41
Epoch  1/15 | Train: 1.4420 (61.61%) | Test: 0.4516 (93.51%) | LR: 0.000708 | Time: 10.5s                                                                                                     
  🏆 New best model saved! (Test Acc: 93.51%)
Epoch  2/15 | Train: 0.2027 (96.16%) | Test: 0.1006 (97.23%) | LR: 0.001460 | Time: 10.0s                                                                                                     
  🏆 New best model saved! (Test Acc: 97.23%)
Epoch  3/15 | Train: 0.0874 (97.63%) | Test: 0.0642 (98.21%) | LR: 0.001487 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.21%)
Epoch  4/15 | Train: 0.0658 (98.07%) | Test: 0.0512 (98.60%) | LR: 0.001431 | Time: 10.0s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.60%)
Epoch  5/15 | Train: 0.0518 (98.49%) | Test: 0.0432 (98.77%) | LR: 0.001334 | Time: 10.0s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.77%)
Epoch  6/15 | Train: 0.0460 (98.65%) | Test: 0.0375 (98.81%) | LR: 0.001202 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.81%)
Epoch  7/15 | Train: 0.0388 (98.83%) | Test: 0.0330 (99.05%) | LR: 0.001042 | Time: 10.0s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.05%)
Epoch  8/15 | Train: 0.0330 (99.03%) | Test: 0.0329 (99.02%) | LR: 0.000865 | Time: 10.0s                                                                                                     
  ⏳ No improvement (1/8)
Epoch  9/15 | Train: 0.0285 (99.15%) | Test: 0.0300 (99.06%) | LR: 0.000681 | Time: 9.9s                                                                                                      
  🏆 New best model saved! (Test Acc: 99.06%)
Epoch 10/15 | Train: 0.0255 (99.20%) | Test: 0.0338 (98.93%) | LR: 0.000501 | Time: 10.0s                                                                                                     
  ⏳ No improvement (1/8)
Epoch 11/15 | Train: 0.0210 (99.39%) | Test: 0.0288 (99.17%) | LR: 0.000336 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.17%)
Epoch 12/15 | Train: 0.0182 (99.47%) | Test: 0.0279 (99.18%) | LR: 0.000196 | Time: 10.0s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.18%)
Epoch 13/15 | Train: 0.0161 (99.52%) | Test: 0.0267 (99.26%) | LR: 0.000089 | Time: 10.0s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.26%)
Epoch 14/15 | Train: 0.0152 (99.54%) | Test: 0.0278 (99.21%) | LR: 0.000023 | Time: 10.0s                                                                                                     
  ⏳ No improvement (1/8)
Epoch 15/15 | Train: 0.0139 (99.60%) | Test: 0.0269 (99.24%) | LR: 0.000000 | Time: 10.1s                                                                                                     
  ⏳ No improvement (2/8)

✅ Training completed!
   Total time: 2.53 minutes
   Best test accuracy: 99.26% (Epoch 13)

🎉 Training completed successfully!
📊 Final Results:
  Best Test Accuracy: 99.26%
  Final Test Accuracy: 99.24%
```

### 📊 Analysis

* Replacing the final convolutional block with **Global Average Pooling (GAP)** reduced the model to **7,954 parameters**, bringing it under the **8K target**.
* Achieved **99% by Epoch 7**, peaked at **99.26% at Epoch 13**.
* However, **training accuracy rose to 99.6% while test plateaued at 99.26%**, showing **overfitting**.
* Met parameter but **failed consistency requirement** (never ≥99.4%).
* **Conclusion:** This model stays within the parameter, but it falls short of the 99.4% test accuracy target and shows slight overfitting, as indicated by the widening gap between training and test performance. As the next step, I will add regularization through augmentation techniques.

---

## **MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling – GAP + Data Augmentation (Final Experiment)**

**Architecture (high-level):**

* Uses **Global Average Pooling (GAP)** to keep parameters low (**7,954**).
* Adds **Data Augmentation** to improve generalization.

### 🛠️ Augmentations Applied

* **RandomAffine**: small rotations (±10°), translations (up to 8%), scaling (95–108%), shear (±8°).
* **RandomPerspective** (25% probability): slight perspective warping for viewpoint robustness.
* **Cutout (RandomErasing)**: erases 2–15% of the image to prevent over-reliance on single features.
* **Normalization**: standard MNIST normalization (mean=0.1307, std=0.3081).

### 📄 Training Logs

**Command To Run the Model**: `python train.py --model gap --epochs 15 --batch_size 64 --scheduler cyclic  --dropout_prob 0.015`

```
🚀 Starting training for 15 epochs...
⏰ Training started at: 2025-09-26 17:25:59
Epoch  1/15 | Train: 1.5838 (53.52%) | Test: 0.5048 (93.50%) | LR: 0.000708 | Time: 10.8s                                                                                                     
  🏆 New best model saved! (Test Acc: 93.50%)
Epoch  2/15 | Train: 0.3352 (92.02%) | Test: 0.0927 (97.53%) | LR: 0.001460 | Time: 10.2s                                                                                                     
  🏆 New best model saved! (Test Acc: 97.53%)
Epoch  3/15 | Train: 0.1654 (95.16%) | Test: 0.0697 (97.98%) | LR: 0.001487 | Time: 10.2s                                                                                                     
  🏆 New best model saved! (Test Acc: 97.98%)
Epoch  4/15 | Train: 0.1319 (95.94%) | Test: 0.0439 (98.62%) | LR: 0.001431 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.62%)
Epoch  5/15 | Train: 0.1149 (96.42%) | Test: 0.0555 (98.22%) | LR: 0.001334 | Time: 10.1s                                                                                                     
  ⏳ No improvement (1/8)
Epoch  6/15 | Train: 0.1041 (96.80%) | Test: 0.0345 (98.87%) | LR: 0.001202 | Time: 10.2s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.87%)
Epoch  7/15 | Train: 0.0957 (97.03%) | Test: 0.0318 (98.95%) | LR: 0.001042 | Time: 10.2s                                                                                                     
  🏆 New best model saved! (Test Acc: 98.95%)
Epoch  8/15 | Train: 0.0902 (97.28%) | Test: 0.0297 (99.05%) | LR: 0.000865 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.05%)
Epoch  9/15 | Train: 0.0800 (97.56%) | Test: 0.0235 (99.33%) | LR: 0.000681 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.33%)
Epoch 10/15 | Train: 0.0749 (97.61%) | Test: 0.0215 (99.31%) | LR: 0.000501 | Time: 10.2s                                                                                                     
  ⏳ No improvement (1/8)
Epoch 11/15 | Train: 0.0733 (97.72%) | Test: 0.0211 (99.37%) | LR: 0.000336 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.37%)
Epoch 12/15 | Train: 0.0669 (97.92%) | Test: 0.0203 (99.42%) | LR: 0.000196 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.42%)
Epoch 13/15 | Train: 0.0666 (97.96%) | Test: 0.0194 (99.41%) | LR: 0.000089 | Time: 10.2s                                                                                                     
  ⏳ No improvement (1/8)
Epoch 14/15 | Train: 0.0651 (98.01%) | Test: 0.0187 (99.41%) | LR: 0.000023 | Time: 10.1s                                                                                                     
  ⏳ No improvement (2/8)
Epoch 15/15 | Train: 0.0619 (98.07%) | Test: 0.0185 (99.46%) | LR: 0.000000 | Time: 10.1s                                                                                                     
  🏆 New best model saved! (Test Acc: 99.46%)

✅ Training completed!
   Total time: 2.56 minutes
   Best test accuracy: 99.46% (Epoch 15)

🎉 Training completed successfully!
📊 Final Results:
  Best Test Accuracy: 99.46%
  Final Test Accuracy: 99.46%
```

### 📊 Analysis

* The architecture of the model is same as the last one, here we have applied different image augmentation techniques. 
* Crossed **99.42% at Epoch 12** and **consistently ≥99.4% thereafter**.
* Best accuracy **99.46% at Epoch 15**.
* Train accuracy slightly lower (~98%) due to harder augmented inputs, but test performance improved — a **sign of good generalization**.
* **Conclusion:** Achieved all three targets — **accuracy, epochs, and parameter**. This is the **final chosen model**.

---

# ✅ Final Outcome

| Model                     | Params    | Best Acc.  | Target Met? | Notes                                           |
| ------------------------- | --------- | ---------- | ----------- | ----------------------------------------------- |
| **MNISTNet4BlockWithBatchNormMaxPoolConvFinal (BatchNorm)**   | >8K       | 98.7%      | ❌           | Stable, under target                            |
| **MNISTNet4BlockWithBatchNormMaxPoolDropoutConvFinal (Dropout)**     | >8K       | 98.95%     | ❌           | Better regularization, still short              |
| **MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling (Vanilla GAP)** | 7,954     | 99.26%     | ❌           | Parameter-efficient but overfit, not consistent |
| **MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling (GAP + Aug)**   | **7,954** | **99.46%** | ✅           | Meets all targets, best generalization          |

---

# 🏆 Conclusion

Through iterative experimentation, we:

1. **Started with BatchNorm CNN** (Good model architecture as the test accuracy hit 99.4% target but it was overfitting and the parameter count was more than 8k).
2. **Tried Dropout CNN** (The paramter count was same as last model but here by applying dropout the overfitting was reduced).
3. **Introduced GAP CNN** (The paramter count was reduced by using global average pooling instead of final convolution layer so the model paramter count was less than 8k, but the test accuracy was lower than the targte and model was still overfitting which suggested to apply other regularization techniques such as image augmentation).
4. **Finalized GAP + Data Augmentations** (This model is the best model so far from the experiments it acheives the test accuracy 99.4 and the model is not overfitting with training of few more epoch could provide better test accuracy).

👉 Final model achieved:

* **99.46% accuracy** within **15 epochs**.
* Consistently ≥ **99.4% from Epoch 12 onwards**.
* **7,954 parameters**, under the 8K parameter.

