"""
ResNet50 Model Architecture Module for ImageNet
Contains ResNet50 implementation for ImageNet-1K classification (1000 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet50/101/152.
    Uses 1x1 -> 3x3 -> 1x1 convolution pattern.
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.0):
        super(Bottleneck, self).__init__()
        
        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv for main processing
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        # Store input for skip connection
        identity = x
        
        # 1x1 conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1x1 conv block (expansion)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply dropout if specified
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Apply downsample to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out


class ResNet50ImageNet(nn.Module):
    """
    ResNet50 architecture for ImageNet-1K classification.
    
    Architecture:
    - Initial Conv: 7x7 conv with 64 channels, stride=2
    - Max Pool: 3x3 max pool, stride=2
    - Layer 1: 3 Bottleneck blocks with 64 channels (56x56)
    - Layer 2: 4 Bottleneck blocks with 128 channels (28x28)
    - Layer 3: 6 Bottleneck blocks with 256 channels (14x14)
    - Layer 4: 3 Bottleneck blocks with 512 channels (7x7)
    - GAP: Global Average Pooling (7x7 -> 1x1)
    - FC: Fully Connected layer (1000 classes)
    
    Total layers: 1 + (3 + 4 + 6 + 3) * 3 = 1 + 48 + 1 = 50 layers
    """
    
    def __init__(self, num_classes=1000, dropout=0.0):
        super(ResNet50ImageNet, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.in_channels = 64
        
        # Initial convolution for ImageNet (7x7 conv with stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers - [3, 4, 6, 3] blocks for ResNet50
        self.layer1 = self._make_layer(64, 3, stride=1, dropout_rate=dropout)
        self.layer2 = self._make_layer(128, 4, stride=2, dropout_rate=dropout)
        self.layer3 = self._make_layer(256, 6, stride=2, dropout_rate=dropout)
        self.layer4 = self._make_layer(512, 3, stride=2, dropout_rate=dropout)
        
        # Global Average Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride=1, dropout_rate=0.0):
        """
        Create a ResNet layer with multiple bottleneck blocks.
        
        Args:
            out_channels: Number of output channels
            num_blocks: Number of bottleneck blocks
            stride: Stride for the first block (default: 1)
            dropout_rate: Dropout rate for regularization
            
        Returns:
            nn.Sequential: Layer with multiple blocks
        """
        downsample = None
        
        # Create downsample layer if needed (when stride != 1 or channel mismatch)
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )
        
        layers = []
        
        # First block (may have downsample)
        layers.append(Bottleneck(self.in_channels, out_channels, stride, 
                                downsample, dropout_rate))
        self.in_channels = out_channels * Bottleneck.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_channels, out_channels, 
                                    dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the ResNet50 network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution and pooling
        x = self.conv1(x)      # 224x224 -> 112x112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 112x112 -> 56x56
        
        # ResNet layers
        x = self.layer1(x)     # 56x56 (64 * 4 = 256 channels)
        x = self.layer2(x)     # 28x28 (128 * 4 = 512 channels)
        x = self.layer3(x)     # 14x14 (256 * 4 = 1024 channels)
        x = self.layer4(x)     # 7x7 (512 * 4 = 2048 channels)
        
        # Global Average Pooling
        x = self.avgpool(x)    # 7x7 -> 1x1
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.fc(x)
        
        return x
    
    def get_parameter_count(self):
        """
        Get the total number of parameters in the model.
        
        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def resnet50(num_classes=1000, dropout=0.0, pretrained=False):
    """
    Create ResNet50 model.
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
        dropout: Dropout rate (default: 0.0)
        pretrained: Load pretrained weights (default: False)
        
    Returns:
        ResNet50ImageNet: ResNet50 model instance
    """
    model = ResNet50ImageNet(num_classes=num_classes, dropout=dropout)
    
    if pretrained:
        # Load pretrained weights from torchvision
        import torchvision.models as models
        pretrained_model = models.resnet50(pretrained=True)
        
        # Copy weights (excluding final FC layer if num_classes != 1000)
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        
        # Filter out FC layer if num_classes is different
        if num_classes != 1000:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if not k.startswith('fc')}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print("✅ Loaded pretrained ResNet50 weights")
    
    return model


def test_model():
    """Test model architecture."""
    print("🔍 Testing ResNet50 Architecture")
    print("=" * 50)
    
    # Create model
    model = resnet50(num_classes=1000, dropout=0.1)
    
    # Test with ImageNet input
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of classes: {output.shape[1]}")
    
    print("✅ Model test completed!")


if __name__ == "__main__":
    test_model()

