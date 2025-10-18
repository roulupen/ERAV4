"""
Data Loading and Augmentation Module for Tiny ImageNet
Handles Tiny ImageNet dataset loading, preprocessing, and augmentation.

Tiny ImageNet Details:
- 200 classes (subset of ImageNet-1K)
- Image size: 64x64
- Training: 100,000 images (500 per class)
- Validation: 10,000 images (50 per class)
- Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
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


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet Dataset wrapper with albumentation support.
    
    Directory structure:
    tiny-imagenet-200/
    ├── train/
    │   ├── n01443537/
    │   │   ├── images/
    │   │   │   ├── n01443537_0.JPEG
    │   │   │   └── ...
    │   │   └── n01443537_boxes.txt
    │   └── ...
    ├── val/
    │   ├── images/
    │   │   ├── val_0.JPEG
    │   │   └── ...
    │   └── val_annotations.txt
    └── test/
        └── images/
    """
    
    def __init__(self, root_dir, split='train', transform=None, limit_samples=None):
        """
        Args:
            root_dir: Root directory of Tiny ImageNet dataset
            split: 'train', 'val', or 'test'
            transform: Albumentation transforms
            limit_samples: Limit number of samples (for testing)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Setup dataset paths and load samples
        self.samples, self.class_to_idx, self.idx_to_class = self._load_dataset(limit_samples)
        
        print(f"  Loaded {len(self.samples)} samples")
        print(f"  Classes: {len(self.class_to_idx)}")
    
    def _load_dataset(self, limit_samples=None):
        """Load dataset based on split."""
        if self.split == 'train':
            return self._load_train_data(limit_samples)
        elif self.split == 'val':
            return self._load_val_data(limit_samples)
        else:
            raise ValueError(f"Split '{self.split}' not supported. Use 'train' or 'val'.")
    
    def _load_train_data(self, limit_samples=None):
        """Load training data from class directories."""
        train_dir = self.root_dir / 'train'
        
        if not train_dir.exists():
            raise ValueError(f"Training directory not found: {train_dir}")
        
        # Get all class directories
        class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        
        # Create class to index mapping
        class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}
        idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
        
        # Load samples
        samples = []
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = class_to_idx[class_name]
            
            # Images are in 'images' subdirectory
            images_dir = class_dir / 'images'
            if not images_dir.exists():
                continue
            
            # Get all JPEG images
            image_files = list(images_dir.glob('*.JPEG'))
            
            for img_path in image_files:
                samples.append((str(img_path), class_idx))
                
                if limit_samples and len(samples) >= limit_samples:
                    return samples, class_to_idx, idx_to_class
        
        return samples, class_to_idx, idx_to_class
    
    def _load_val_data(self, limit_samples=None):
        """Load validation data using annotations file."""
        val_dir = self.root_dir / 'val'
        images_dir = val_dir / 'images'
        annotations_file = val_dir / 'val_annotations.txt'
        
        if not images_dir.exists():
            raise ValueError(f"Validation images directory not found: {images_dir}")
        
        if not annotations_file.exists():
            raise ValueError(f"Validation annotations file not found: {annotations_file}")
        
        # Parse annotations file
        # Format: filename	class_id	x	y	width	height
        image_to_class = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    filename = parts[0]
                    class_id = parts[1]
                    image_to_class[filename] = class_id
        
        # Get unique classes and create mapping
        unique_classes = sorted(set(image_to_class.values()))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
        
        # Load samples
        samples = []
        for img_file in sorted(images_dir.glob('*.JPEG')):
            filename = img_file.name
            if filename in image_to_class:
                class_name = image_to_class[filename]
                class_idx = class_to_idx[class_name]
                samples.append((str(img_file), class_idx))
                
                if limit_samples and len(samples) >= limit_samples:
                    break
        
        return samples, class_to_idx, idx_to_class
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            # Return a blank image
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def get_tinyimagenet_transforms(augment=True, mean=None, std=None, augment_strength='medium'):
    """
    Get Tiny ImageNet transforms with albumentation augmentation.
    Optimized for 64x64 images.
    
    Args:
        augment: Whether to apply augmentation (default: True)
        mean: Dataset mean for normalization (default: ImageNet mean)
        std: Dataset std for normalization (default: ImageNet std)
        augment_strength: Augmentation strength - 'light', 'medium', or 'strong' (default: 'medium')
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Use ImageNet normalization values
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    if augment:
        if augment_strength == 'light':
            # Light augmentation - minimal transformations
            train_transform = A.Compose([
                A.RandomResizedCrop(height=64, width=64, scale=(0.85, 1.0), ratio=(0.95, 1.05), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        
        elif augment_strength == 'strong':
            # Strong augmentation - maximum diversity
            train_transform = A.Compose([
                A.RandomResizedCrop(height=64, width=64, scale=(0.6, 1.0), ratio=(0.85, 1.15), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.7),
                
                # Strong color augmentation
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                ], p=0.9),
                
                # Multiple noise/blur options
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.4),
                
                # Advanced augmentations
                A.OneOf([
                    A.CoarseDropout(max_holes=12, max_height=10, max_width=10, 
                                   min_holes=1, min_height=4, min_width=4, fill_value=0, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
                    A.OpticalDistortion(distort_limit=0.15, shift_limit=0.15, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=10, p=1.0),
                ], p=0.4),
                
                # Pixel-level
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=1.0),
                    A.Sharpen(alpha=(0.2, 0.6), lightness=(0.5, 1.0), p=1.0),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                ], p=0.5),
                
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        
        else:  # medium (default)
            # Medium augmentation - balanced approach (RECOMMENDED)
            train_transform = A.Compose([
                # Geometric augmentations
                A.RandomResizedCrop(height=64, width=64, scale=(0.75, 1.0), ratio=(0.9, 1.1), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,  # 4 pixels at 64x64
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    border_mode=0,
                    p=0.6
                ),
                
                # Color augmentations
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
                ], p=0.8),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.3),
                
                # Advanced augmentations
                A.OneOf([
                    A.CoarseDropout(
                        max_holes=8, 
                        max_height=8, 
                        max_width=8, 
                        min_holes=1,
                        min_height=4,
                        min_width=4,
                        fill_value=0,
                        p=1.0
                    ),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                ], p=0.3),
                
                # Pixel-level augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                ], p=0.4),
                
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
    else:
        # No augmentation for training
        train_transform = A.Compose([
            A.Resize(height=64, width=64),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(height=64, width=64),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def get_tinyimagenet_data_loaders(
    data_dir,
    batch_size=128,
    num_workers=4,
    augment=True,
    augment_strength='medium',
    pin_memory=None,
    limit_samples=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get Tiny ImageNet data loaders with albumentation augmentation.
    
    Args:
        data_dir: Root directory of Tiny ImageNet dataset
        batch_size: Batch size for data loaders (default: 128)
        num_workers: Number of worker processes (default: 4)
        augment: Whether to apply augmentation (default: True)
        augment_strength: Augmentation strength - 'light', 'medium', or 'strong' (default: 'medium')
        pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        limit_samples: Limit number of samples for testing (default: None)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Auto-detect pin_memory based on device
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    print(f"\n📥 Loading Tiny ImageNet dataset from: {data_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Augmentation: {'✅' if augment else '❌'} ({augment_strength})")
    print(f"  Pin memory: {'✅' if pin_memory else '❌'}")
    if limit_samples:
        print(f"  ⚠️  Limiting to {limit_samples} samples (test mode)")
    
    # Get transforms
    train_transform, val_transform = get_tinyimagenet_transforms(
        augment=augment,
        augment_strength=augment_strength
    )
    
    # Create datasets
    print(f"\n📚 Loading Training Set...")
    train_dataset = TinyImageNetDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        limit_samples=limit_samples
    )
    
    print(f"\n📚 Loading Validation Set...")
    val_dataset = TinyImageNetDataset(
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


def get_tinyimagenet_data_loaders_limited(
    data_dir,
    train_samples=20000,
    batch_size=128,
    num_workers=4,
    augment=True,
    augment_strength='medium',
    pin_memory=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get Tiny ImageNet data loaders with LIMITED training samples but FULL validation set.
    Useful for pipeline testing and debugging.
    
    Args:
        data_dir: Root directory of Tiny ImageNet dataset
        train_samples: Number of training samples to use (default: 20,000)
        batch_size: Batch size for data loaders (default: 128)
        num_workers: Number of worker processes (default: 4)
        augment: Whether to apply augmentation (default: True)
        augment_strength: Augmentation strength - 'light', 'medium', or 'strong' (default: 'medium')
        pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Auto-detect pin_memory based on device
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    print(f"\n📥 Loading Tiny ImageNet dataset (LIMITED TRAINING MODE)")
    print(f"  Data directory: {data_dir}")
    print(f"  Training samples: {train_samples:,} (limited for testing)")
    print(f"  Validation samples: ALL (~10,000)")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Augmentation: {'✅' if augment else '❌'} ({augment_strength})")
    print(f"  Pin memory: {'✅' if pin_memory else '❌'}")
    
    # Get transforms
    train_transform, val_transform = get_tinyimagenet_transforms(
        augment=augment,
        augment_strength=augment_strength
    )
    
    # Create datasets
    print(f"\n📚 Loading Training Set...")
    train_dataset = TinyImageNetDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        limit_samples=train_samples  # LIMIT training samples
    )
    
    print(f"\n📚 Loading Validation Set...")
    val_dataset = TinyImageNetDataset(
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


def visualize_samples(
    data_loader,
    num_samples=10,
    dataset_name='Dataset',
    class_names=None,
    figsize=(20, 8)
):
    """
    Visualize sample images from a data loader.
    
    Args:
        data_loader: PyTorch DataLoader to visualize samples from
        num_samples: Number of samples to display (default: 10)
        dataset_name: Name of the dataset for title (default: 'Dataset')
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


def get_tinyimagenet_class_names(data_dir):
    """
    Get Tiny ImageNet class names (synset IDs).
    
    Args:
        data_dir: Root directory of Tiny ImageNet dataset
        
    Returns:
        list: List of class names (synset IDs)
    """
    train_dir = Path(data_dir) / 'train'
    
    if train_dir.exists():
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        return class_names
    else:
        return None


def get_dataset_info():
    """
    Get Tiny ImageNet dataset information.
    
    Returns:
        dict: Dataset information
    """
    return {
        'name': 'Tiny ImageNet',
        'num_classes': 200,
        'input_size': (3, 64, 64),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'train_samples': 100000,
        'val_samples': 10000,
        'images_per_class_train': 500,
        'images_per_class_val': 50,
        'download_url': 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    }


def test_data_loading(data_dir):
    """
    Test data loading functionality.
    
    Args:
        data_dir: Root directory of Tiny ImageNet dataset
    """
    print("🔍 Testing Tiny ImageNet Data Loading")
    print("=" * 50)
    
    try:
        # Test with small batch and limited samples
        train_loader, val_loader = get_tinyimagenet_data_loaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            augment=True,
            limit_samples=100
        )
        
        # Test a batch
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Data shape: {data.shape}")
            print(f"  Target shape: {target.shape}")
            print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
            print(f"  Target range: [{target.min()}, {target.max()}]")
            break
        
        print("\n✅ Data loading test completed!")
        
    except Exception as e:
        print(f"\n❌ Error during data loading test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Display dataset info
    info = get_dataset_info()
    print("\n📋 Tiny ImageNet Dataset Information:")
    print("=" * 50)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with a sample directory (update path as needed)
    test_dir = "./data/tiny-imagenet-200"
    if not os.path.exists(test_dir):
        print(f"\n⚠️  Test directory not found: {test_dir}")
        print("Please download Tiny ImageNet from:")
        print(info['download_url'])
    else:
        test_data_loading(test_dir)

