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
    CIFAR-10 Network with C1C2C3C40 architecture.
    
    Architecture:
    - C1: 3x3 Conv + BatchNorm + ReLU
    - C2: 3x3 Depthwise Separable Conv + BatchNorm + ReLU  
    - C3: 3x3 Dilated Conv (dilation=2) + BatchNorm + ReLU
    - C40: 3x3 Conv with stride=2 + BatchNorm + ReLU
    - GAP: Global Average Pooling
    - FC: Fully Connected layer (optional)
    
    Total parameters: < 200k
    Receptive Field: > 44
    """
    
    def __init__(self, num_classes=10, use_fc=True):
        super(CIFAR10Net, self).__init__()
        
        self.num_classes = num_classes
        self.use_fc = use_fc
        
        # C1: Initial convolution - very small channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # C2: Depthwise Separable Convolution - small channels
        self.conv2 = DepthwiseSeparableConv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # C3: Dilated Convolution (dilation=8 for increased RF) - small channels
        self.conv3 = DilatedConv2d(32, 64, kernel_size=3, padding=8, dilation=8)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Additional dilated layer for more RF
        self.conv3b = DilatedConv2d(64, 64, kernel_size=3, padding=16, dilation=16)
        self.bn3b = nn.BatchNorm2d(64)
        
        # C40: Final convolution with stride=2 (replaces MaxPooling) - small channels
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Optional FC layer
        if use_fc:
            self.fc = nn.Linear(128, num_classes)
        else:
            # If no FC, we need to adjust the final conv to output num_classes
            self.conv_final = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # C1: 32x32 -> 32x32
        x = F.relu(self.bn1(self.conv1(x)))
        
        # C2: 32x32 -> 32x32 (Depthwise Separable)
        x = F.relu(self.bn2(self.conv2(x)))
        
        # C3: 32x32 -> 32x32 (Dilated)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # C3b: 32x32 -> 32x32 (Additional Dilated)
        x = F.relu(self.bn3b(self.conv3b(x)))
        
        # C40: 32x32 -> 16x16 (Stride=2)
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global Average Pooling: 16x16 -> 1x1
        x = self.gap(x)
        
        if self.use_fc:
            # Flatten and FC
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
        else:
            # Use 1x1 conv instead of FC
            x = self.conv_final(x)
            x = x.view(x.size(0), -1)
        
        return F.log_softmax(x, dim=1)
    
    def calculate_receptive_field(self):
        """
        Calculate the receptive field of the network.
        Returns the receptive field size.
        """
        # RF calculation for each layer
        rf = 1
        stride = 1
        
        # C1: 3x3 conv, padding=1
        rf = rf + (3-1) * stride
        stride = stride * 1  # stride=1
        
        # C2: 3x3 depthwise separable, padding=1  
        rf = rf + (3-1) * stride
        stride = stride * 1  # stride=1
        
        # C3: 3x3 dilated conv, dilation=8, padding=8
        effective_kernel = 3 + (3-1) * (8-1)  # 17x17 effective kernel
        rf = rf + (effective_kernel-1) * stride
        stride = stride * 1  # stride=1
        
        # C3b: 3x3 dilated conv, dilation=16, padding=16
        effective_kernel = 3 + (3-1) * (16-1)  # 33x33 effective kernel
        rf = rf + (effective_kernel-1) * stride
        stride = stride * 1  # stride=1
        
        # C40: 3x3 conv, stride=2, padding=1
        rf = rf + (3-1) * stride
        stride = stride * 2  # stride=2
        
        return rf
    
    def get_parameter_count(self):
        """
        Get the total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cifar10_model(num_classes=10, use_fc=True):
    """
    Create a CIFAR-10 model instance.
    
    Args:
        num_classes: Number of output classes (default: 10)
        use_fc: Whether to use FC layer after GAP (default: True)
        
    Returns:
        CIFAR10Net: Model instance
    """
    return CIFAR10Net(num_classes=num_classes, use_fc=use_fc)


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
    print(f"Receptive field: {model.calculate_receptive_field()}")
    print(f"Under 200k params: {'✅' if model.get_parameter_count() < 200000 else '❌'}")
    print(f"RF > 44: {'✅' if model.calculate_receptive_field() > 44 else '❌'}")
    
    # Test without FC layer
    model_no_fc = create_cifar10_model(use_fc=False)
    output_no_fc = model_no_fc(x)
    print(f"\nWithout FC layer:")
    print(f"Parameter count: {model_no_fc.get_parameter_count():,}")
    print(f"Output shape: {output_no_fc.shape}")
    
    return model


if __name__ == "__main__":
    test_model_architecture()
