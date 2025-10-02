"""
CIFAR-10 Model Architecture Module
Contains the C1C2C3C40 architecture with Depthwise Separable Convolution,
Dilated Convolution, and Global Average Pooling.

Receptive Field Progression:
Input (32x32): RF = 1
├── Block 1 (3x DepthwiseSeparable 3x3): RF = 1 → 3 → 5 → 7
├── Transition 1 (1x1): RF = 7
├── Block 2 (3x DepthwiseSeparable 3x3): RF = 7 → 9 → 11 → 13
├── Transition 2 (1x1): RF = 13
├── Block 3 (3x Dilated 3x3, dilation=2): RF = 13 → 17 → 21 → 25
├── Transition 3 (1x1): RF = 25
├── Block 4 (3x Dilated 3x3, dilation=4): RF = 25 → 33 → 41 → 49
├── Transition 4 (1x1, stride=2): RF = 49
├── GAP: RF = 49
└── FC: Final RF = 49 ✅ (> 44)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution implementation.
    Separates spatial and channel-wise convolutions for efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,
            groups=in_channels
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DilatedConv2d(nn.Module):
    """
    Dilated Convolution implementation for increased receptive field.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DilatedConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation
        )
        
    def forward(self, x):
        return self.conv(x)


class CIFAR10Net(nn.Module):
    """
    CIFAR-10 Network with 4-Block Architecture using Depthwise and Dilated Convolutions.
    
    Architecture:
    - Block 1: 3x3 Depthwise Separable Conv (RF: 1->7)
    - Block 2: 3x3 Depthwise Separable Conv (RF: 7->13)
    - Block 3: 3x3 Dilated Conv (dilation=2) (RF: 13->25)
    - Block 4: 3x3 Dilated Conv (dilation=4) (RF: 25->49)
    - Transitions: 1x1 Conv for channel reduction
    - Final Transition: 1x1 Conv with stride=2 for downsampling
    - GAP: Global Average Pooling
    - FC: Fully Connected layer
    
    Total parameters: < 200k
    Final Receptive Field: 49 (> 44 ✅)
    """
    
    def __init__(self, num_classes=10, dropout=0.05):
        super(CIFAR10Net, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Block 1: Multiple convolution layers (32, 64, 128 channels)
        # Input: 32x32, RF: 1
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv2d(3, 32, kernel_size=3, padding=1),    # RF: 3 (3x3 conv)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1),   # RF: 5 (3x3 conv)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1),  # RF: 7 (3x3 conv)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Transition 1: Reduce channels to 32
        self.transition1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),                            # RF: 7 (1x1 conv, no change)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: Depthwise Separable Convolution (32, 64, 128 channels)
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv2d(32, 32, kernel_size=3, padding=1),   # RF: 9 (3x3 conv)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1),   # RF: 11 (3x3 conv)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1),  # RF: 13 (3x3 conv)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Transition 2: Reduce channels to 32
        self.transition2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),                            # RF: 13 (1x1 conv, no change)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: Dilated Convolution (dilation=2) (32, 64, 128 channels)
        self.block3 = nn.Sequential(
            DilatedConv2d(32, 32, kernel_size=3, padding=2, dilation=2), # RF: 17 (3x3 dilated, eff_k=5)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DilatedConv2d(32, 64, kernel_size=3, padding=2, dilation=2), # RF: 21 (3x3 dilated, eff_k=5)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DilatedConv2d(64, 128, kernel_size=3, padding=2, dilation=2),# RF: 25 (3x3 dilated, eff_k=5)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Transition 3: Reduce channels to 16
        self.transition3 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1),                            # RF: 25 (1x1 conv, no change)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: Dilated Convolution (dilation=4) (16, 32, 64 channels)
        self.block4 = nn.Sequential(
            DilatedConv2d(16, 32, kernel_size=3, padding=4, dilation=4), # RF: 33 (3x3 dilated, eff_k=9)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DilatedConv2d(32, 64, kernel_size=3, padding=4, dilation=4), # RF: 41 (3x3 dilated, eff_k=9)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DilatedConv2d(64, 64, kernel_size=3, padding=4, dilation=4), # RF: 49 (3x3 dilated, eff_k=9)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(dropout)
        )
        
        # Transition 4: Reduce channels to 16 and downsample
        self.transition4 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=2),                   # RF: 49 (1x1 conv, stride=2)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)                                # RF: 49 (GAP, no change)
        
        # Final classification layer
        self.fc = nn.Linear(16, num_classes)                              # Final RF: 49
        
    def forward(self, x):
        # Block 1: 32x32 -> 32x32 (32, 64, 128 channels), RF: 1->7
        x = self.block1(x)
        
        # Transition 1: 32x32 -> 32x32 (128 -> 32 channels), RF: 7
        x = self.transition1(x)
        
        # Block 2: 32x32 -> 32x32 (Depthwise Separable, 32, 64, 128 channels), RF: 7->13
        x = self.block2(x)
        
        # Transition 2: 32x32 -> 32x32 (128 -> 32 channels), RF: 13
        x = self.transition2(x)
        
        # Block 3: 32x32 -> 32x32 (Dilated, dilation=2, 32, 64, 128 channels), RF: 13->25
        x = self.block3(x)
        
        # Transition 3: 32x32 -> 32x32 (128 -> 16 channels), RF: 25
        x = self.transition3(x)
        
        # Block 4: 32x32 -> 32x32 (Dilated, dilation=4, 16, 32, 64 channels), RF: 25->49
        x = self.block4(x)
        
        # Transition 4: 32x32 -> 16x16 (Downsampling, 64 -> 16 channels), RF: 49
        x = self.transition4(x)
        
        # Global Average Pooling: 16x16 -> 1x1, RF: 49
        x = self.gap(x)
        
        # Flatten and FC (no dropout on last layer), Final RF: 49
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    

def create_cifar10_model(num_classes=10, dropout=0.05):
    """
    Create a CIFAR-10 model instance.
    
    Args:
        num_classes: Number of output classes (default: 10)
        dropout: Dropout rate for all dropout layers (default: 0.05)
        
    Returns:
        CIFAR10Net: Model instance
    """
    return CIFAR10Net(num_classes=num_classes, dropout=dropout)
