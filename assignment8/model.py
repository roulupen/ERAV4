"""
ResNet18 Model Architecture Module for CIFAR-100
Contains ResNet18 implementation adapted for CIFAR-100 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    """
    Basic ResNet block for ResNet18/34.
    Uses Sequential blocks for cleaner organization.
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.1):
        super(BasicBlock, self).__init__()
        
        # First conv block: conv1 + bn1 + relu + dropout
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Second conv block: conv2 + bn2 + dropout (no relu here - applied after skip connection)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate)
        )
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        # Store input for skip connection (residual connection)
        skip_connection = x
        
        # First conv block: conv1 + bn1 + relu + dropout
        out = self.conv_block1(x)
        
        # Second conv block: conv2 + bn2 + dropout (no relu here)
        out = self.conv_block2(out)
        
        # Apply downsample to skip connection if needed (when stride > 1 or channel mismatch)
        if self.downsample is not None:
            skip_connection = self.downsample(x)
            
        # Add residual connection: F(x) + x
        out += skip_connection
        
        # Apply ReLU after skip connection (standard ResNet pattern)
        out = F.relu(out)
        
        return out


class ResNet18CIFAR100(nn.Module):
    """
    ResNet18 architecture adapted for CIFAR-100.
    
    Key modifications for CIFAR-100:
    - Smaller initial kernel (3x3 instead of 7x7)
    - No initial max pooling
    - Adjusted channel dimensions
    - Global Average Pooling
    - Proper weight initialization
    - 100 output classes for CIFAR-100
    
    Architecture:
    - Initial Conv: 3x3 conv with 64 channels
    - Layer 1: 2 BasicBlocks with 64 channels (32x32)
    - Layer 2: 2 BasicBlocks with 128 channels (16x16)
    - Layer 3: 2 BasicBlocks with 256 channels (8x8)
    - Layer 4: 2 BasicBlocks with 512 channels (4x4)
    - GAP: Global Average Pooling (4x4 -> 1x1)
    - FC: Fully Connected layer (100 classes)
    """
    
    def __init__(self, num_classes=100, dropout=0.1):
        super(ResNet18CIFAR100, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.in_channels = 64
        
        # Initial convolution (adapted for CIFAR-100) with dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # ResNet layers - Direct BasicBlock usage with dropout (much simpler!)
        
        # Layer 1: 64 channels, 32x32 (no downsample needed)
        self.layer1_block1 = BasicBlock(64, 64, stride=1, dropout_rate=dropout)
        self.layer1_block2 = BasicBlock(64, 64, stride=1, dropout_rate=dropout)
        
        # Layer 2: 128 channels, 16x16 (downsample needed for first block)
        # Skip connection downsample: 1x1 conv + bn + dropout to match main path
        self.layer2_downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout)
        )
        self.layer2_block1 = BasicBlock(64, 128, stride=2, downsample=self.layer2_downsample, dropout_rate=dropout)
        self.layer2_block2 = BasicBlock(128, 128, stride=1, dropout_rate=dropout)
        
        # Layer 3: 256 channels, 8x8 (downsample needed for first block)
        # Skip connection downsample: 1x1 conv + bn + dropout to match main path
        self.layer3_downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout)
        )
        self.layer3_block1 = BasicBlock(128, 256, stride=2, downsample=self.layer3_downsample, dropout_rate=dropout)
        self.layer3_block2 = BasicBlock(256, 256, stride=1, dropout_rate=dropout)
        
        # Layer 4: 512 channels, 4x4 (downsample needed for first block)
        # Skip connection downsample: 1x1 conv + bn + dropout to match main path
        self.layer4_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(dropout)
        )
        self.layer4_block1 = BasicBlock(256, 512, stride=2, downsample=self.layer4_downsample, dropout_rate=dropout)
        self.layer4_block2 = BasicBlock(512, 512, stride=1, dropout_rate=dropout)

        # Global Average Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
    
    def forward(self, x):
        """
        Forward pass through the ResNet18 network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution (now includes dropout)
        x = self.conv1(x)  # 32x32 -> 32x32 (conv + bn + relu + dropout)
        
        # ResNet layers - Direct BasicBlock usage with skip connections
        
        # Layer 1: 64 channels, 32x32 (no downsample needed - same channels and size)
        x = self.layer1_block1(x)  # 32x32 -> 32x32 (skip connection: identity)
        x = self.layer1_block2(x)  # 32x32 -> 32x32 (skip connection: identity)
        
        # Layer 2: 128 channels, 16x16 (downsample in first block - channel and size change)
        x = self.layer2_block1(x)  # 32x32 -> 16x16 (skip connection: 1x1 conv + stride=2)
        x = self.layer2_block2(x)  # 16x16 -> 16x16 (skip connection: identity)
        
        # Layer 3: 256 channels, 8x8 (downsample in first block - channel and size change)
        x = self.layer3_block1(x)  # 16x16 -> 8x8 (skip connection: 1x1 conv + stride=2)
        x = self.layer3_block2(x)  # 8x8 -> 8x8 (skip connection: identity)
        
        # Layer 4: 512 channels, 4x4 (downsample in first block - channel and size change)
        x = self.layer4_block1(x)  # 8x8 -> 4x4 (skip connection: 1x1 conv + stride=2)
        x = self.layer4_block2(x)  # 4x4 -> 4x4 (skip connection: identity)
        
        # Global Average Pooling
        x = self.avgpool(x)  # 4x4 -> 1x1
        x = torch.flatten(x, 1)
        
        # Classifier with dropout for regularization
        #x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_parameter_count(self):
        """
        Get the total number of parameters in the model.
        
        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNet34CIFAR100(nn.Module):
    """
    ResNet34 architecture adapted for CIFAR-100.
    
    Key modifications for CIFAR-100:
    - Smaller initial kernel (3x3 instead of 7x7)
    - No initial max pooling
    - Adjusted channel dimensions
    - Global Average Pooling
    - Proper weight initialization
    - 100 output classes for CIFAR-100
    
    Architecture:
    - Initial Conv: 3x3 conv with 64 channels
    - Layer 1: 3 BasicBlocks with 64 channels (32x32)
    - Layer 2: 4 BasicBlocks with 128 channels (16x16)
    - Layer 3: 6 BasicBlocks with 256 channels (8x8)
    - Layer 4: 3 BasicBlocks with 512 channels (4x4)
    - GAP: Global Average Pooling (4x4 -> 1x1)
    - FC: Fully Connected layer (100 classes)
    
    Total layers: 1 + 3 + 4 + 6 + 3 = 17 blocks = 34 layers (including conv layers)
    """
    
    def __init__(self, num_classes=100, dropout=0.1):
        super(ResNet34CIFAR100, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.in_channels = 64
        
        # Initial convolution (adapted for CIFAR-100) with dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # ResNet layers - ResNet34 has 3, 4, 6, 3 blocks per layer
        
        # Layer 1: 64 channels, 32x32 (3 blocks, no downsample needed for first block)
        self.layer1_block1 = BasicBlock(64, 64, stride=1, dropout_rate=dropout)
        self.layer1_block2 = BasicBlock(64, 64, stride=1, dropout_rate=dropout)
        self.layer1_block3 = BasicBlock(64, 64, stride=1, dropout_rate=dropout)
        
        # Layer 2: 128 channels, 16x16 (4 blocks, downsample needed for first block)
        # Skip connection downsample: 1x1 conv + bn + dropout to match main path
        self.layer2_downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout)
        )
        self.layer2_block1 = BasicBlock(64, 128, stride=2, downsample=self.layer2_downsample, dropout_rate=dropout)
        self.layer2_block2 = BasicBlock(128, 128, stride=1, dropout_rate=dropout)
        self.layer2_block3 = BasicBlock(128, 128, stride=1, dropout_rate=dropout)
        self.layer2_block4 = BasicBlock(128, 128, stride=1, dropout_rate=dropout)
        
        # Layer 3: 256 channels, 8x8 (6 blocks, downsample needed for first block)
        # Skip connection downsample: 1x1 conv + bn + dropout to match main path
        self.layer3_downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout)
        )
        self.layer3_block1 = BasicBlock(128, 256, stride=2, downsample=self.layer3_downsample, dropout_rate=dropout)
        self.layer3_block2 = BasicBlock(256, 256, stride=1, dropout_rate=dropout)
        self.layer3_block3 = BasicBlock(256, 256, stride=1, dropout_rate=dropout)
        self.layer3_block4 = BasicBlock(256, 256, stride=1, dropout_rate=dropout)
        self.layer3_block5 = BasicBlock(256, 256, stride=1, dropout_rate=dropout)
        self.layer3_block6 = BasicBlock(256, 256, stride=1, dropout_rate=dropout)
        
        # Layer 4: 512 channels, 4x4 (3 blocks, downsample needed for first block)
        # Skip connection downsample: 1x1 conv + bn + dropout to match main path
        self.layer4_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(dropout)
        )
        self.layer4_block1 = BasicBlock(256, 512, stride=2, downsample=self.layer4_downsample, dropout_rate=dropout)
        self.layer4_block2 = BasicBlock(512, 512, stride=1, dropout_rate=dropout)
        self.layer4_block3 = BasicBlock(512, 512, stride=1, dropout_rate=dropout)

        # Global Average Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the ResNet34 network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution (now includes dropout)
        x = self.conv1(x)  # 32x32 -> 32x32 (conv + bn + relu + dropout)
        
        # ResNet layers - ResNet34 has 3, 4, 6, 3 blocks per layer
        
        # Layer 1: 64 channels, 32x32 (3 blocks, no downsample needed)
        x = self.layer1_block1(x)  # 32x32 -> 32x32 (skip connection: identity)
        x = self.layer1_block2(x)  # 32x32 -> 32x32 (skip connection: identity)
        x = self.layer1_block3(x)  # 32x32 -> 32x32 (skip connection: identity)
        
        # Layer 2: 128 channels, 16x16 (4 blocks, downsample in first block)
        x = self.layer2_block1(x)  # 32x32 -> 16x16 (skip connection: 1x1 conv + stride=2)
        x = self.layer2_block2(x)  # 16x16 -> 16x16 (skip connection: identity)
        x = self.layer2_block3(x)  # 16x16 -> 16x16 (skip connection: identity)
        x = self.layer2_block4(x)  # 16x16 -> 16x16 (skip connection: identity)
        
        # Layer 3: 256 channels, 8x8 (6 blocks, downsample in first block)
        x = self.layer3_block1(x)  # 16x16 -> 8x8 (skip connection: 1x1 conv + stride=2)
        x = self.layer3_block2(x)  # 8x8 -> 8x8 (skip connection: identity)
        x = self.layer3_block3(x)  # 8x8 -> 8x8 (skip connection: identity)
        x = self.layer3_block4(x)  # 8x8 -> 8x8 (skip connection: identity)
        x = self.layer3_block5(x)  # 8x8 -> 8x8 (skip connection: identity)
        x = self.layer3_block6(x)  # 8x8 -> 8x8 (skip connection: identity)
        
        # Layer 4: 512 channels, 4x4 (3 blocks, downsample in first block)
        x = self.layer4_block1(x)  # 8x8 -> 4x4 (skip connection: 1x1 conv + stride=2)
        x = self.layer4_block2(x)  # 4x4 -> 4x4 (skip connection: identity)
        x = self.layer4_block3(x)  # 4x4 -> 4x4 (skip connection: identity)
        
        # Global Average Pooling
        x = self.avgpool(x)  # 4x4 -> 1x1
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_parameter_count(self):
        """
        Get the total number of parameters in the model.
        
        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

