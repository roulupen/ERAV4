"""
Assignment 8: CIFAR-100 Classification with ResNet18

This module contains CIFAR-100 specific implementations:
- ResNet18 model adapted for CIFAR-100 (100 classes)
- CIFAR-100 data loading and augmentation
"""

from .model import ResNet18CIFAR100, ResNet34CIFAR100
from .data import get_cifar100_data_loaders, get_dataset_info

__all__ = [
    'ResNet18CIFAR100',
    'ResNet34CIFAR100',
    'get_cifar100_data_loaders',
    'get_dataset_info',
]

__version__ = '1.0.0'

