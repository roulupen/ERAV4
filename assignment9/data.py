"""
Data Loading and Augmentation Module for ImageNet
Handles ImageNet dataset loading, preprocessing, and augmentation.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt


class ImageNetDataset(Dataset):
    """
    ImageNet Dataset wrapper with albumentation support.
    Handles both Kaggle and local ImageNet directory structures.
    """
    
    def __init__(self, root_dir, split='train', transform=None, limit_samples=None):
        """
        Args:
            root_dir: Root directory of ImageNet dataset
            split: 'train' or 'val'
            transform: Albumentation transforms
            limit_samples: Limit number of samples (for testing)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Detect directory structure (Kaggle vs local)
        self.image_dir, self.class_to_idx, self.val_annotations = self._setup_dataset()
        self.samples = self._load_samples(limit_samples)
        
    def _setup_dataset(self):
        """Setup dataset paths based on directory structure."""
        # Try Kaggle structure first: /kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/
        kaggle_base = Path(self.root_dir) / 'ILSVRC' / 'Data' / 'CLS-LOC'
        val_annotations = None
        
        if kaggle_base.exists():
            print(f"📁 Detected Kaggle ImageNet structure")
            if self.split == 'train':
                image_dir = kaggle_base / 'train'
            else:
                image_dir = kaggle_base / 'val'
                # Load validation annotations for Kaggle structure
                val_annotations = self._load_kaggle_val_annotations()
        else:
            # Standard ImageNet structure
            print(f"📁 Using standard ImageNet structure")
            image_dir = Path(self.root_dir) / self.split
        
        # Get class folders
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        # Create class to index mapping
        if self.split == 'val' and val_annotations is not None:
            # For Kaggle validation, use the synsets from annotations
            class_folders = sorted(set(val_annotations.values()))
        else:
            # For training or standard structure, use directory structure
            class_folders = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
        
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}
        
        print(f"  Found {len(class_to_idx)} classes")
        
        return image_dir, class_to_idx, val_annotations
    
    def _load_kaggle_val_annotations(self):
        """
        Load validation annotations for Kaggle ImageNet structure.
        The validation images are in a flat directory and need to be mapped using annotations.
        """
        annotations_path = Path(self.root_dir) / 'LOC_val_solution.csv'
        
        if not annotations_path.exists():
            print(f"⚠️  Validation annotations not found at: {annotations_path}")
            print(f"  Trying alternative annotation locations...")
            
            # Try alternative paths
            alt_paths = [
                Path(self.root_dir) / 'ILSVRC' / 'Annotations' / 'CLS-LOC' / 'val',
                Path(self.root_dir) / 'ILSVRC' / 'ImageSets' / 'CLS-LOC' / 'val.txt',
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    print(f"  Found alternative annotation path: {alt_path}")
                    annotations_path = alt_path
                    break
        
        # Parse the validation annotations
        val_annotations = {}
        
        if annotations_path.exists() and annotations_path.suffix == '.csv':
            # CSV format: ImageId,PredictionString
            # PredictionString format: class_label x1 y1 x2 y2
            import csv
            with open(annotations_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_id = row['ImageId']
                    # Extract class label (first part before coordinates)
                    pred_string = row['PredictionString']
                    class_label = pred_string.split()[0] if pred_string else None
                    if class_label:
                        val_annotations[img_id] = class_label
        else:
            # Fall back: try to infer from XML annotations or use directory structure
            print(f"⚠️  No CSV annotations found. Checking for XML annotations...")
            annotations_dir = Path(self.root_dir) / 'ILSVRC' / 'Annotations' / 'CLS-LOC' / 'val'
            
            if annotations_dir.exists():
                import xml.etree.ElementTree as ET
                for xml_file in annotations_dir.glob('*.xml'):
                    try:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()
                        img_name = xml_file.stem
                        
                        # Extract class from first object
                        obj = root.find('object')
                        if obj is not None:
                            class_name = obj.find('name').text
                            val_annotations[img_name] = class_name
                    except Exception as e:
                        continue
        
        print(f"  Loaded {len(val_annotations)} validation annotations")
        return val_annotations if val_annotations else None
    
    def _load_samples(self, limit_samples=None):
        """Load image paths and labels."""
        samples = []
        
        # Handle Kaggle validation structure (flat directory with annotations)
        if self.split == 'val' and self.val_annotations is not None:
            # Validation images are in a flat directory
            # Get all images from the flat validation directory
            image_files = list(self.image_dir.glob('*.JPEG')) + list(self.image_dir.glob('*.jpg'))
            
            for img_path in image_files:
                img_name = img_path.stem  # Get filename without extension
                
                # Get class label from annotations
                if img_name in self.val_annotations:
                    class_name = self.val_annotations[img_name]
                    if class_name in self.class_to_idx:
                        class_idx = self.class_to_idx[class_name]
                        samples.append((str(img_path), class_idx))
                        
                        if limit_samples and len(samples) >= limit_samples:
                            break
        else:
            # Training or standard structure: images organized in class directories
            for class_name, class_idx in self.class_to_idx.items():
                class_dir = self.image_dir / class_name
                
                if not class_dir.exists():
                    continue
                
                # Get all images in class directory
                image_files = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg'))
                
                for img_path in image_files:
                    samples.append((str(img_path), class_idx))
                    
                    if limit_samples and len(samples) >= limit_samples:
                        break
                
                if limit_samples and len(samples) >= limit_samples:
                    break
        
        print(f"  Loaded {len(samples)} samples")
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


def get_imagenet_transforms(augment=True, mean=None, std=None):
    """
    Get ImageNet transforms with albumentation augmentation.
    
    Args:
        augment: Whether to apply augmentation (default: True)
        mean: Dataset mean for normalization (default: ImageNet mean)
        std: Dataset std for normalization (default: ImageNet std)
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    # ImageNet normalization values
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet std
    
    if augment:
        # Training transforms with strong augmentation
        train_transform = A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(p=1.0),
            ], p=0.3),
            # Cutout augmentation for improved regularization
            A.CoarseDropout(
                max_holes=1,           # typically 1–2 holes work best
                max_height=56,         # ~25% of 224
                max_width=56,
                min_holes=1,
                min_height=32,
                min_width=32,
                fill_value=0,
                p=0.6
            ),
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


def get_imagenet_data_loaders(
    data_dir,
    batch_size=128,
    num_workers=4,
    augment=True,
    pin_memory=None,
    limit_samples=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet data loaders with albumentation augmentation.
    
    Args:
        data_dir: Root directory of ImageNet dataset
        batch_size: Batch size for data loaders (default: 128)
        num_workers: Number of worker processes (default: 4)
        augment: Whether to apply augmentation (default: True)
        pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        limit_samples: Limit number of samples for testing (default: None)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Auto-detect pin_memory based on device
    if pin_memory is None:
        if torch.cuda.is_available():
            pin_memory = True
        else:
            pin_memory = False
    
    print(f"\n📥 Loading ImageNet dataset from: {data_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Augmentation: {'✅' if augment else '❌'}")
    print(f"  Pin memory: {'✅' if pin_memory else '❌'}")
    if limit_samples:
        print(f"  ⚠️  Limiting to {limit_samples} samples (test mode)")
    
    # Get transforms
    train_transform, val_transform = get_imagenet_transforms(augment=augment)
    
    # Create datasets
    train_dataset = ImageNetDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        limit_samples=limit_samples
    )
    
    val_dataset = ImageNetDataset(
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
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def get_imagenet_data_loaders_limited(
    data_dir,
    train_samples=200000,
    batch_size=128,
    num_workers=4,
    augment=True,
    pin_memory=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet data loaders with LIMITED training samples but FULL validation set.
    Useful for pipeline testing and debugging before full training.
    
    Args:
        data_dir: Root directory of ImageNet dataset
        train_samples: Number of training samples to use (default: 200,000)
        batch_size: Batch size for data loaders (default: 128)
        num_workers: Number of worker processes (default: 4)
        augment: Whether to apply augmentation (default: True)
        pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Auto-detect pin_memory based on device
    if pin_memory is None:
        if torch.cuda.is_available():
            pin_memory = True
        else:
            pin_memory = False
    
    print(f"\n📥 Loading ImageNet dataset (LIMITED TRAINING MODE)")
    print(f"  Data directory: {data_dir}")
    print(f"  Training samples: {train_samples:,} (limited for testing)")
    print(f"  Validation samples: ALL (~50,000)")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Augmentation: {'✅' if augment else '❌'}")
    print(f"  Pin memory: {'✅' if pin_memory else '❌'}")
    
    # Get transforms
    train_transform, val_transform = get_imagenet_transforms(augment=augment)
    
    # Create datasets
    train_dataset = ImageNetDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        limit_samples=train_samples  # LIMIT training samples
    )
    
    val_dataset = ImageNetDataset(
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
    Get ImageNet dataset information.
    
    Returns:
        dict: Dataset information
    """
    return {
        'name': 'ImageNet-1K',
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'train_samples': 1281167,
        'val_samples': 50000
    }


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


def get_imagenet_class_names(data_dir):
    """
    Get ImageNet class names (synset IDs).
    
    Args:
        data_dir: Root directory of ImageNet dataset
        
    Returns:
        list: List of class names (synset IDs)
    """
    # Try Kaggle structure first
    kaggle_base = Path(data_dir) / 'ILSVRC' / 'Data' / 'CLS-LOC'
    
    if kaggle_base.exists():
        train_dir = kaggle_base / 'train'
    else:
        train_dir = Path(data_dir) / 'train'
    
    if train_dir.exists():
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        return class_names
    else:
        return None


def test_data_loading(data_dir):
    """
    Test data loading functionality.
    
    Args:
        data_dir: Root directory of ImageNet dataset
    """
    print("🔍 Testing ImageNet Data Loading")
    print("=" * 50)
    
    try:
        # Test with small batch and limited samples
        train_loader, val_loader = get_imagenet_data_loaders(
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
    # Test with a sample directory
    test_dir = "/kaggle/input/imagenet-object-localization-challenge"
    if not os.path.exists(test_dir):
        print(f"⚠️  Test directory not found: {test_dir}")
        print("Please provide a valid ImageNet directory path")
    else:
        test_data_loading(test_dir)

