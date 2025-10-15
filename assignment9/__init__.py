"""
Assignment 9: ResNet50 ImageNet Training

This package contains ResNet50 implementation for ImageNet-1K classification.
"""

from assignment9.model import ResNet50ImageNet, resnet50
from assignment9.data import get_imagenet_data_loaders, ImageNetDataset
from assignment9.config import TrainingConfig, get_default_config

__version__ = '1.0.0'
__all__ = [
    'ResNet50ImageNet',
    'resnet50',
    'get_imagenet_data_loaders',
    'ImageNetDataset',
    'TrainingConfig',
    'get_default_config',
]

