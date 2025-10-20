"""
Assignment 9: ResNet50 ImageNet Training

This package contains ResNet50 implementation for ImageNet-1K classification.

Note: Import specific modules directly to avoid dependency conflicts:
    from ERAV4.assignment9.model import resnet50
    from ERAV4.assignment9.data_imagenet100 import get_imagenet100_data_loaders
"""

__version__ = '1.0.0'

# Lazy imports to avoid scipy/numpy conflicts in Kaggle
# Import specific modules directly instead of using __init__.py
__all__ = [
    'ResNet50ImageNet',
    'resnet50',
    'get_imagenet_data_loaders',
    'ImageNetDataset',
    'TrainingConfig',
    'get_default_config',
]

def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies until needed"""
    if name == 'resnet50' or name == 'ResNet50ImageNet':
        from ERAV4.assignment9.model import resnet50, ResNet50ImageNet
        return resnet50 if name == 'resnet50' else ResNet50ImageNet
    elif name == 'get_imagenet_data_loaders' or name == 'ImageNetDataset':
        from ERAV4.assignment9.data import get_imagenet_data_loaders, ImageNetDataset
        return get_imagenet_data_loaders if name == 'get_imagenet_data_loaders' else ImageNetDataset
    elif name == 'TrainingConfig' or name == 'get_default_config':
        from ERAV4.assignment9.config import TrainingConfig, get_default_config
        return TrainingConfig if name == 'TrainingConfig' else get_default_config
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

