"""
Data loading and visualization utilities for MNIST training pipeline.
Contains functions for loading MNIST dataset and visualizing data.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def get_mnist_data_loaders(batch_size=64, num_workers=4, data_dir='./data', pin_memory=False, cutout_prob=0.2):
    """
    Create data loaders for MNIST training and testing with appropriate transforms.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory to store MNIST data
        pin_memory: Whether to pin memory for faster GPU transfer
        cutout_prob: Probability of applying Cutout augmentation (0.0 to 1.0)
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,                    # rotate +/- 10°
            translate=(0.08, 0.08),        # up to 8% translation
            scale=(0.95, 1.08),            # slight scale
            shear=8,                       # shear up to 8°
            fill=0                         # fill background with 0 (black) - set to 1 if your background is white
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.25),  # small perspective warp
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # Cutout augmentation: randomly erase rectangular regions
        transforms.RandomErasing(p=cutout_prob, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ])
    
    # Simple transform for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"📊 Data loaded successfully:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, test_loader


def visualize_mnist_samples(data_loader, num_samples=16, title="MNIST Training Samples", figsize=(10, 10)):
    """
    Visualize sample images from the dataset.
    
    Args:
        data_loader: DataLoader containing the dataset
        num_samples: Number of samples to display
        title: Title for the plot
        figsize: Figure size tuple
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Denormalize the image for visualization
        img = images[i].squeeze()
        img = img * 0.3081 + 0.1307  # Reverse normalization
        img = torch.clamp(img, 0, 1)  # Clamp to valid range
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, data_loader, device, num_samples=16, figsize=(12, 12)):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: The trained model
        data_loader: DataLoader containing test data
        device: Device to run inference on
        num_samples: Number of samples to display
        figsize: Figure size tuple
    """
    model.eval()
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        probabilities = torch.softmax(outputs, dim=1)
        confidence = probabilities.max(dim=1)[0]
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Denormalize the image for visualization
        img = images[i].cpu().squeeze()
        img = img * 0.3081 + 0.1307  # Reverse normalization
        img = torch.clamp(img, 0, 1)  # Clamp to valid range
        
        axes[i].imshow(img, cmap='gray')
        
        pred = predictions[i].item()
        true_label = labels[i].item()
        conf = confidence[i].item() * 100
        
        # Color code: green for correct, red for incorrect
        color = 'green' if pred == true_label else 'red'
        
        axes[i].set_title(f'True: {true_label}\nPred: {pred} ({conf:.1f}%)', 
                         color=color, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions on Test Samples', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, data_loader, device, figsize=(10, 8)):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model: The trained model
        data_loader: DataLoader containing test data
        device: Device to run inference on
        figsize: Figure size tuple
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    from tqdm import tqdm
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Generating predictions"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=[str(i) for i in range(10)]))


def plot_training_history(training_history, save_path='./training_history.png', figsize=(15, 10)):
    """
    Plot and save training history.
    
    Args:
        training_history: Dictionary containing training metrics
        save_path: Path to save the plot
        figsize: Figure size tuple
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    epochs = training_history['epochs']
    
    # Loss plot
    ax1.plot(epochs, training_history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, training_history['test_losses'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, training_history['train_accuracies'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, training_history['test_accuracies'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax3.plot(epochs, training_history['learning_rates'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Accuracy difference plot
    acc_diff = [t - test for t, test in zip(training_history['train_accuracies'], training_history['test_accuracies'])]
    ax4.plot(epochs, acc_diff, 'purple', linewidth=2)
    ax4.set_title('Train-Test Accuracy Difference', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training history plot saved to: {save_path}")


def get_cutout_data_loaders(batch_size=64, num_workers=4, data_dir='./data', pin_memory=False, cutout_intensity='medium'):
    """
    Create data loaders with different Cutout augmentation intensities.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory to store MNIST data
        pin_memory: Whether to pin memory for faster GPU transfer
        cutout_intensity: Cutout intensity level ('light', 'medium', 'heavy')
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define Cutout parameters based on intensity
    cutout_configs = {
        'light': {'p': 0.1, 'scale': (0.01, 0.08), 'ratio': (0.3, 3.3)},
        'medium': {'p': 0.2, 'scale': (0.02, 0.15), 'ratio': (0.3, 3.3)},
        'heavy': {'p': 0.3, 'scale': (0.03, 0.25), 'ratio': (0.3, 3.3)}
    }
    
    config = cutout_configs.get(cutout_intensity, cutout_configs['medium'])
    
    # Training transform with configurable Cutout
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.08, 0.08),
            scale=(0.95, 1.08),
            shear=8,
            fill=0
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # Configurable Cutout augmentation
        transforms.RandomErasing(
            p=config['p'], 
            scale=config['scale'], 
            ratio=config['ratio'], 
            value=0
        )
    ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"📊 Cutout data loaded successfully:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Cutout intensity: {cutout_intensity}")
    print(f"  Cutout probability: {config['p']}")
    print(f"  Cutout scale: {config['scale']}")
    
    return train_loader, test_loader
