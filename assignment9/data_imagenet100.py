"""
Data Loading and Augmentation Module for ImageNet-100
Handles ImageNet-100 dataset loading, preprocessing, and augmentation.

ImageNet-100 is a subset of ImageNet with 100 classes, providing a good balance
between training speed and model performance evaluation.

Dataset: https://www.kaggle.com/datasets/wilyzh/imagenet100/data
"""

import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt


class ImageNet100Dataset(Dataset):
    """
    ImageNet-100 Dataset wrapper with albumentation support.
    
    ImageNet-100 contains 100 classes from ImageNet:
    - Training: ~126,689 images
    - Validation: ~5,000 images (50 per class)
    """
    
    def __init__(self, root_dir, split='train', transform=None, limit_samples=None):
        """
        Args:
            root_dir: Root directory of ImageNet-100 dataset
            split: 'train' or 'val'
            transform: Albumentation transforms
            limit_samples: Limit number of samples (for testing)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Setup dataset paths
        self.image_dir, self.class_to_idx = self._setup_dataset()
        self.samples = self._load_samples(limit_samples)
        
        print(f"  Loaded {len(self.samples)} samples from {split} set")
        print(f"  Classes: {len(self.class_to_idx)}")
        
    def _setup_dataset(self):
        """Setup dataset paths and class mappings."""
        # ImageNet-100 typical structure:
        # imagenet100/
        #   ├── train/
        #   │   ├── class1/
        #   │   ├── class2/
        #   │   └── ...
        #   └── val/
        #       ├── class1/
        #       ├── class2/
        #       └── ...
        
        # Try different possible directory structures
        possible_paths = [
            self.root_dir / 'ImageNet100' / self.split,   # Kaggle: imagenet100/ImageNet100/train
            self.root_dir / self.split,  # Direct: imagenet100/train or imagenet100/val
            self.root_dir / 'ImageNet-100' / self.split,  # Alternative: imagenet100/ImageNet-100/train
            self.root_dir / 'imagenet100' / self.split,   # Lowercase: imagenet100/imagenet100/train
            self.root_dir / f'train.X1' if self.split == 'train' else self.root_dir / f'val.X1',  # Kaggle format with .X1
            self.root_dir / f'train.X' if self.split == 'train' else self.root_dir / f'val.X',    # Kaggle format with .X
            self.root_dir / 'Training_Set' if self.split == 'train' else self.root_dir / 'Validation_Set',  # Alternative naming
        ]
        
        image_dir = None
        
        # First, show what actually exists in root_dir for debugging
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
        
        print(f"🔍 Contents of {self.root_dir}:")
        for item in sorted(self.root_dir.iterdir())[:20]:  # Show first 20 items
            print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")
        
        # Try to find the correct path
        for path in possible_paths:
            if path.exists():
                image_dir = path
                print(f"✅ Found ImageNet-100 {self.split} directory: {path}")
                break
        
        # If still not found, try auto-detection by looking for directories with ~100 class subdirectories
        if image_dir is None:
            print(f"⚠️  Standard paths not found, trying auto-detection...")
            for item in self.root_dir.iterdir():
                if item.is_dir():
                    subdirs = [d for d in item.iterdir() if d.is_dir()]
                    # Check if this directory contains ~100 class folders (90-110 range to be flexible)
                    if 90 <= len(subdirs) <= 110:
                        # Check if the subdirectory names look like ImageNet class IDs (start with 'n')
                        sample_names = [d.name for d in subdirs[:5]]
                        if all(name.startswith('n') and len(name) >= 8 for name in sample_names):
                            image_dir = item
                            print(f"✅ Auto-detected {self.split} directory: {item} ({len(subdirs)} classes)")
                            break
        
        if image_dir is None:
            raise ValueError(
                f"Could not find {self.split} directory. Tried:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths) +
                f"\n\nActual contents shown above. Please check the dataset structure."
            )
        
        # Get class folders (should be 100 classes)
        class_folders = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
        
        if len(class_folders) != 100:
            print(f"⚠️  Warning: Expected 100 classes, found {len(class_folders)}")
        
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}
        
        return image_dir, class_to_idx
    
    def _load_samples(self, limit_samples=None):
        """Load image paths and labels."""
        samples = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.image_dir / class_name
            
            if not class_dir.exists():
                continue
            
            # Get all images in class directory
            image_files = (
                list(class_dir.glob('*.JPEG')) + 
                list(class_dir.glob('*.jpg')) + 
                list(class_dir.glob('*.png'))
            )
            
            for img_path in image_files:
                samples.append((str(img_path), class_idx))
                
                if limit_samples and len(samples) >= limit_samples:
                    break
            
            if limit_samples and len(samples) >= limit_samples:
                break
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def get_imagenet100_transforms(augment=True, augment_strength='strong', mean=None, std=None):
    """
    Get ImageNet-100 transforms with albumentation augmentation.
    
    Args:
        augment: Whether to apply augmentation (default: True)
        augment_strength: 'light', 'medium', or 'strong' (default: 'strong')
        mean: Dataset mean for normalization (default: ImageNet mean)
        std: Dataset std for normalization (default: ImageNet std)
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    # ImageNet normalization values (same for ImageNet-100)
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    if augment:
        if augment_strength == 'light':
            # Light augmentation - basic transforms
            train_transform = A.Compose([
                A.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        
        elif augment_strength == 'medium':
            # Medium augmentation - balanced approach
            train_transform = A.Compose([
                A.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0),
                ], p=0.7),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(p=1.0),
                ], p=0.3),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=48,
                    max_width=48,
                    min_holes=1,
                    min_height=28,
                    min_width=28,
                    fill_value=0,
                    p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        
        else:  # strong (default)
            # Strong augmentation - maximum diversity
            train_transform = A.Compose([
                A.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                
                # Strong color augmentation
                A.OneOf([
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                ], p=0.8),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], p=0.4),
                
                # Cutout augmentation
                A.CoarseDropout(
                    max_holes=1,
                    max_height=56,
                    max_width=56,
                    min_holes=1,
                    min_height=32,
                    min_width=32,
                    fill_value=0,
                    p=0.6
                ),
                
                # Optional advanced augmentations
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.OpticalDistortion(distort_limit=0.1, p=1.0),
                ], p=0.2),
                
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
    else:
        # No augmentation for training
        train_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def get_imagenet100_data_loaders(
    data_dir,
    batch_size=128,
    num_workers=4,
    augment=True,
    augment_strength='strong',
    pin_memory=None,
    limit_samples=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet-100 data loaders with albumentation augmentation.
    
    Args:
        data_dir: Root directory of ImageNet-100 dataset
        batch_size: Batch size for data loaders (default: 128)
        num_workers: Number of worker processes (default: 4)
        augment: Whether to apply augmentation (default: True)
        augment_strength: 'light', 'medium', or 'strong' (default: 'strong')
        pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        limit_samples: Limit number of samples for testing (default: None)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Auto-detect pin_memory based on device
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    print(f"\n📥 Loading ImageNet-100 dataset from: {data_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Augmentation: {'✅' if augment else '❌'} ({augment_strength})")
    print(f"  Pin memory: {'✅' if pin_memory else '❌'}")
    if limit_samples:
        print(f"  ⚠️  Limiting to {limit_samples} samples (test mode)")
    
    # Get transforms
    train_transform, val_transform = get_imagenet100_transforms(
        augment=augment,
        augment_strength=augment_strength
    )
    
    # Create datasets
    print(f"\n📚 Loading Training Set...")
    train_dataset = ImageNet100Dataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        limit_samples=limit_samples
    )
    
    print(f"\n📚 Loading Validation Set...")
    val_dataset = ImageNet100Dataset(
        root_dir=data_dir,
        split='val',
        transform=val_transform,
        limit_samples=limit_samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\n✅ Data loaders ready!")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Train samples: ~{len(train_dataset):,}")
    print(f"  Val samples: ~{len(val_dataset):,}")
    
    return train_loader, val_loader


def get_imagenet100_data_loaders_limited(
    data_dir,
    train_samples=50000,
    batch_size=128,
    num_workers=4,
    augment=True,
    augment_strength='strong',
    pin_memory=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet-100 data loaders with LIMITED training samples but FULL validation set.
    Useful for pipeline testing and debugging before full training.
    
    Args:
        data_dir: Root directory of ImageNet-100 dataset
        train_samples: Number of training samples to use (default: 50,000)
        batch_size: Batch size for data loaders (default: 128)
        num_workers: Number of worker processes (default: 4)
        augment: Whether to apply augmentation (default: True)
        augment_strength: 'light', 'medium', or 'strong' (default: 'strong')
        pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Auto-detect pin_memory based on device
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    print(f"\n📥 Loading ImageNet-100 dataset (LIMITED TRAINING MODE)")
    print(f"  Data directory: {data_dir}")
    print(f"  Training samples: {train_samples:,} (limited for testing)")
    print(f"  Validation samples: ALL (~5,000)")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Augmentation: {'✅' if augment else '❌'} ({augment_strength})")
    print(f"  Pin memory: {'✅' if pin_memory else '❌'}")
    
    # Get transforms
    train_transform, val_transform = get_imagenet100_transforms(
        augment=augment,
        augment_strength=augment_strength
    )
    
    # Create datasets
    print(f"\n📚 Loading Training Set (limited)...")
    train_dataset = ImageNet100Dataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        limit_samples=train_samples  # LIMIT training samples
    )
    
    print(f"\n📚 Loading Validation Set (full)...")
    val_dataset = ImageNet100Dataset(
        root_dir=data_dir,
        split='val',
        transform=val_transform,
        limit_samples=None  # Use ALL validation samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\n✅ Data loaders ready!")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Train samples: ~{len(train_dataset):,}")
    print(f"  Val samples: ~{len(val_dataset):,}")
    
    return train_loader, val_loader


def get_dataset_info():
    """
    Get ImageNet-100 dataset information.
    
    Returns:
        dict: Dataset information
    """
    return {
        'name': 'ImageNet-100',
        'num_classes': 100,
        'input_size': (3, 224, 224),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'train_samples': 126689,  # Approximate
        'val_samples': 5000       # 50 per class
    }


def visualize_samples(
    data_loader,
    num_samples=10,
    dataset_name='ImageNet-100',
    class_names=None,
    figsize=(20, 8)
):
    """
    Visualize sample images from a data loader.
    
    Args:
        data_loader: PyTorch DataLoader to visualize samples from
        num_samples: Number of samples to display (default: 10)
        dataset_name: Name of the dataset for title (default: 'ImageNet-100')
        class_names: Optional list of class names for labels
        figsize: Figure size (default: (20, 8))
    """
    # Get ImageNet normalization values for denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Get a batch of images
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Limit to num_samples
    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.ravel()
    
    for idx in range(num_samples):
        # Get image and denormalize
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean  # Denormalize
        img = np.clip(img, 0, 1)  # Clip to [0, 1] range
        
        # Get label
        label = labels[idx].item()
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Set title with class name or label
        if class_names and label < len(class_names):
            title = f"{class_names[label]}\n(ID: {label})"
        else:
            title = f"Class: {label}"
        axes[idx].set_title(title, fontsize=10)
    
    # Overall title
    fig.suptitle(f'Sample Images from {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Displayed {num_samples} sample images from {dataset_name}")


def get_imagenet100_class_names(data_dir):
    """
    Get ImageNet-100 class names (synset IDs).
    
    Args:
        data_dir: Root directory of ImageNet-100 dataset
        
    Returns:
        list: List of class names (synset IDs)
    """
    possible_paths = [
        Path(data_dir) / 'train',
        Path(data_dir) / 'ImageNet-100' / 'train',
        Path(data_dir) / 'imagenet100' / 'train',
    ]
    
    for train_dir in possible_paths:
        if train_dir.exists():
            class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            return class_names
    
    return None


def test_data_loading(data_dir):
    """
    Test data loading functionality.
    
    Args:
        data_dir: Root directory of ImageNet-100 dataset
    """
    print("🔍 Testing ImageNet-100 Data Loading")
    print("=" * 50)
    
    try:
        # Test with small batch and limited samples
        train_loader, val_loader = get_imagenet100_data_loaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            augment=True,
            augment_strength='medium',
            limit_samples=100
        )
        
        # Test a batch
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Data shape: {data.shape}")
            print(f"  Target shape: {target.shape}")
            print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
            print(f"  Target range: [{target.min()}, {target.max()}]")
            
            # Check if we have 100 classes
            unique_classes = len(torch.unique(target))
            print(f"  Unique classes in batch: {unique_classes}")
            break
        
        print("\n✅ Data loading test completed!")
        print(f"   Expected: 100 classes total")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"\n❌ Error during data loading test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test with a sample directory
    test_dir = "/kaggle/input/imagenet100"
    
    if not os.path.exists(test_dir):
        print(f"⚠️  Test directory not found: {test_dir}")
        print("Please provide a valid ImageNet-100 directory path")
        print("\nExpected directory structure:")
        print("imagenet100/")
        print("  ├── train/")
        print("  │   ├── class1/")
        print("  │   └── ...")
        print("  └── val/")
        print("      ├── class1/")
        print("      └── ...")
    else:
        test_data_loading(test_dir)

