"""
Data Loading and Augmentation Module for CIFAR-100
Handles dataset loading, preprocessing, and albumentation-based augmentation.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Tuple, Optional


class CIFAR100Dataset(torch.utils.data.Dataset):
    """
    CIFAR-100 Dataset wrapper with albumentation support.
    """
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert PIL image to numpy array
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def get_cifar100_transforms(augment=True, mean=None, std=None):
    """
    Get CIFAR-100 transforms with albumentation augmentation.
    
    Args:
        augment: Whether to apply augmentation (default: True)
        mean: Dataset mean for normalization (default: CIFAR-100 mean)
        std: Dataset std for normalization (default: CIFAR-100 std)
        
    Returns:
        tuple: (train_transform, test_transform)
    """
    # CIFAR-100 normalization values
    if mean is None:
        mean = [0.5071, 0.4867, 0.4408]  # CIFAR-100 mean
    if std is None:
        std = [0.2675, 0.2565, 0.2761]   # CIFAR-100 std
    
    if augment:
        # Training transforms with Albumentations
        train_transform = A.Compose([
            A.PadIfNeeded(min_height=36, min_width=36, p=1.0), 
            A.RandomCrop(32, 32),                        # standard CIFAR trick (pad+crop)
            A.HorizontalFlip(p=0.5),                     # flip
            A.Affine(                                    # replaces deprecated ShiftScaleRotate
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5
            ),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5), # brightness/contrast
            A.HueSaturationValue(20, 30, 20, p=0.5),     # color jitter
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(16, 16),
                hole_width_range=(16, 16),
                p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        # No augmentation for training
        train_transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    # Test transforms (no augmentation)
    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, test_transform


def get_cifar100_data_loaders(
    batch_size=32,
    num_workers=4,
    data_dir='./data',
    augment=True,
    pin_memory=None,
    mean=None,
    std=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders with albumentation augmentation.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        data_dir: Directory to store CIFAR-100 data
        augment: Whether to apply augmentation
        pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        mean: Dataset mean for normalization
        std: Dataset std for normalization
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Auto-detect pin_memory based on device
    if pin_memory is None:
        import torch
        if torch.cuda.is_available():
            pin_memory = True
        elif torch.backends.mps.is_available():
            pin_memory = False  # MPS doesn't support pin_memory
        else:
            pin_memory = False
    # Get transforms
    train_transform, test_transform = get_cifar100_transforms(
        augment=augment, mean=mean, std=std
    )
    
    # Load CIFAR-100 datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=None  # We'll apply transforms in our wrapper
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=None  # We'll apply transforms in our wrapper
    )
    
    # Wrap with our custom dataset class
    train_dataset = CIFAR100Dataset(train_dataset, transform=train_transform)
    test_dataset = CIFAR100Dataset(test_dataset, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader


def get_dataset_info():
    """
    Get CIFAR-100 dataset information.
    
    Returns:
        dict: Dataset information
    """
    return {
        'name': 'CIFAR-100',
        'num_classes': 100,
        'input_size': (3, 32, 32),
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'train_samples': 50000,
        'test_samples': 10000
    }


def visualize_augmentations(dataset, num_samples=8, save_path=None):
    """
    Visualize data augmentations.
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to show
        save_path: Path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    
    # Get samples
    indices = torch.randperm(len(dataset))[:num_samples]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            image = image * std + mean
            image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f'Class: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Augmentation visualization saved to: {save_path}")
    
    plt.show()


def test_data_loading():
    """
    Test data loading functionality.
    """
    print("🔍 Testing CIFAR-100 Data Loading")
    print("=" * 40)
    
    # Test with augmentation
    train_loader, test_loader = get_cifar100_data_loaders(
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        augment=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Target values: {target.unique()}")
        break
    
    # Test without augmentation
    train_loader_no_aug, _ = get_cifar100_data_loaders(
        batch_size=4,
        num_workers=0,
        augment=False
    )
    
    print(f"\nWithout augmentation:")
    for batch_idx, (data, target) in enumerate(train_loader_no_aug):
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
        break
    
    print("✅ Data loading test completed!")


if __name__ == "__main__":
    test_data_loading()

