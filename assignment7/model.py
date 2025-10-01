"""
CIFAR-10 Model Architecture Module
Contains the C1C2C3C40 architecture with Depthwise Separable Convolution,
Dilated Convolution, and Global Average Pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    - Block 1: 3x3 Conv + BatchNorm + ReLU + Dropout
    - Block 2: 3x3 Depthwise Separable Conv + BatchNorm + ReLU + Dropout
    - Block 3: 3x3 Dilated Conv (dilation=2) + BatchNorm + ReLU + Dropout
    - Block 4: 3x3 Dilated Conv (dilation=4) + BatchNorm + ReLU + Dropout
    - Transition: 1x1 Conv with stride=2 for downsampling
    - GAP: Global Average Pooling
    - FC: Fully Connected layer (optional)
    
    Total parameters: < 200k
    Receptive Field: > 44
    """
    
    def __init__(self, num_classes=10, dropout=0.05):
        super(CIFAR10Net, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Block 1: Multiple convolution layers (16, 32, 64 channels)
        self.block1 = nn.Sequential(
            #nn.Conv2d(3, 32, kernel_size=3, padding=1),
            DepthwiseSeparableConv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            #nn.Conv2d(32, 64, kernel_size=3, padding=1),
            DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            #nn.Conv2d(64, 128, kernel_size=3, padding=1),
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Transition 1: Reduce channels to 16
        self.transition1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: Depthwise Separable Convolution (16, 32, 64 channels)
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Transition 2: Reduce channels to 16
        self.transition2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: Dilated Convolution (dilation=2) (16, 32, 64 channels)
        self.block3 = nn.Sequential(
            DilatedConv2d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DilatedConv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DilatedConv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Transition 3: Reduce channels to 16
        self.transition3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: Dilated Convolution (dilation=4) (16, 32, 64 channels)
        self.block4 = nn.Sequential(
            DilatedConv2d(32, 32, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DilatedConv2d(32, 32, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            DilatedConv2d(32, 32, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(dropout)
        )
        
        # Transition 4: Reduce channels to 16 and downsample
        self.transition4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # Block 1: 32x32 -> 32x32 (16, 32, 64 channels)
        x = self.block1(x)
        
        # Transition 1: 32x32 -> 32x32 (64 -> 16 channels)
        x = self.transition1(x)
        
        # Block 2: 32x32 -> 32x32 (Depthwise Separable, 16, 32, 64 channels)
        x = self.block2(x)
        
        # Transition 2: 32x32 -> 32x32 (64 -> 16 channels)
        x = self.transition2(x)
        
        # Block 3: 32x32 -> 32x32 (Dilated, dilation=2, 16, 32, 64 channels)
        x = self.block3(x)
        
        # Transition 3: 32x32 -> 32x32 (64 -> 16 channels)
        x = self.transition3(x)
        
        # Block 4: 32x32 -> 32x32 (Dilated, dilation=4, 16, 32, 64 channels)
        x = self.block4(x)
        
        # Transition 4: 32x32 -> 16x16 (Downsampling, 64 -> 16 channels)
        x = self.transition4(x)
        
        # Global Average Pooling: 16x16 -> 1x1
        x = self.gap(x)
        
        # Flatten and FC (no dropout on last layer)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    
    def get_parameter_count(self):
        """
        Get the total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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


def test_model_architecture():
    """
    Test the model architecture and print information.
    """
    model = create_cifar10_model()
    
    print("🔍 CIFAR-10 Model Architecture Test")
    print("=" * 50)
    
    # Test with CIFAR-10 input size
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {model.get_parameter_count():,}")
    print(f"Under 200k params: {'✅' if model.get_parameter_count() < 200000 else '❌'}")
    print(f"Uses GAP: {'✅' if hasattr(model, 'gap') else '❌'}")
    print(f"Uses Depthwise Separable: {'✅' if hasattr(model, 'block2') and 'DepthwiseSeparableConv2d' in str(type(model.block2[0])) else '❌'}")
    print(f"Uses Dilated Conv: {'✅' if hasattr(model, 'block3') and 'DilatedConv2d' in str(type(model.block3[0])) else '❌'}")
    print(f"Has 4 Blocks: {'✅' if hasattr(model, 'block1') and hasattr(model, 'block2') and hasattr(model, 'block3') and hasattr(model, 'block4') else '❌'}")
    print(f"Has Transition Blocks: {'✅' if hasattr(model, 'transition1') and hasattr(model, 'transition2') and hasattr(model, 'transition3') and hasattr(model, 'transition4') else '❌'}")
    
    # Test with different dropout rates
    model_dropout = create_cifar10_model(dropout=0.1)
    output_dropout = model_dropout(x)
    print(f"\nWith dropout=0.1:")
    print(f"Parameter count: {model_dropout.get_parameter_count():,}")
    print(f"Output shape: {output_dropout.shape}")
    
    return model


if __name__ == "__main__":
    test_model_architecture()
