"""
Generic Utility Functions
Contains device detection, random seed setting, checkpointing, and other common operations.
"""

import torch
import random
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt

try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    TORCHSUMMARY_AVAILABLE = False
    print("⚠️  torchsummary not available. Model summary will be limited.")


def set_random_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 Random seed set to {seed}")


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU (Metal Performance Shaders)")
        return device
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")
        return device


def get_model_summary(model, input_size):
    """
    Prints a summary of the model using torchsummary, ensuring the input tensor
    is on the same device as the model.
    """
    TORCHSUMMARY_AVAILABLE = True
    try:
        import torchsummary
    except ImportError:
        TORCHSUMMARY_AVAILABLE = False

    if TORCHSUMMARY_AVAILABLE:
        # Handle MPS device by temporarily moving model to CPU for summary
        device_type = next(model.parameters()).device.type
        original_device = next(model.parameters()).device
        
        if device_type == 'mps':
            # Temporarily move model to CPU for summary
            model_cpu = model.cpu()
            summary(model_cpu, input_size=input_size, device='cpu')
            # Move model back to original device
            model.to(original_device)
        else:
            summary(model, input_size=input_size, device=device_type)
    else:
        # Fallback: just show parameter count
        print("torchsummary not available. Install it for detailed summary.")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {param_count:,}")


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath, scheduler=None, additional_info=None):
    """
    Save model checkpoint with comprehensive information.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        filepath: Path to save the checkpoint
        scheduler: Learning rate scheduler (optional)
        additional_info: Additional information to save (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': time.time()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info is not None:
        checkpoint['additional_info'] = additional_info
    
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, scheduler=None, device=None):
    """
    Load model checkpoint.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        filepath: Path to the checkpoint file
        scheduler: Learning rate scheduler (optional)
        device: Device to load checkpoint on (optional)
        
    Returns:
        dict: Checkpoint data
    """
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def plot_training_history(history, save_path=None, show_plot=True):
    """
    Plot training history with losses and accuracies.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot (default: True)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history.get('epochs', range(1, len(history['train_losses']) + 1))
    
    # Plot training and test losses
    axes[0, 0].plot(epochs, history['train_losses'], label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, history['test_losses'], label='Test Loss', color='red')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training and test accuracies
    axes[0, 1].plot(epochs, history['train_accuracies'], label='Train Accuracy', color='blue')
    axes[0, 1].plot(epochs, history['test_accuracies'], label='Test Accuracy', color='red')
    axes[0, 1].set_title('Training and Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate
    if 'learning_rates' in history:
        axes[1, 0].plot(epochs, history['learning_rates'], label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
    
    # Plot accuracy difference (overfitting indicator)
    if len(history['train_accuracies']) == len(history['test_accuracies']):
        acc_diff = [train - test for train, test in zip(history['train_accuracies'], history['test_accuracies'])]
        axes[1, 1].plot(epochs, acc_diff, label='Train - Test Accuracy', color='purple')
        axes[1, 1].set_title('Overfitting Indicator (Train - Test Accuracy)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_training_info(info, filepath):
    """
    Save training information to JSON file.
    
    Args:
        info: Dictionary containing training information
        filepath: Path to save the information
    """
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Training info saved to: {filepath}")


def load_training_info(filepath):
    """
    Load training information from JSON file.
    
    Args:
        filepath: Path to the information file
        
    Returns:
        dict: Training information
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_accuracy(outputs, targets):
    """
    Calculate accuracy given predictions and targets.
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        
    Returns:
        float: Accuracy percentage
    """
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100.0 * correct / total


def format_time(seconds):
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_training_summary(history, model, config=None):
    """
    Print a comprehensive training summary.
    
    Args:
        history: Training history dictionary
        model: Trained model
        config: Training configuration (optional)
    """
    print("\n" + "="*60)
    print("🎉 TRAINING SUMMARY")
    print("="*60)
    
    # Model information
    if hasattr(model, 'get_parameter_count'):
        param_count = model.get_parameter_count()
        print(f"📊 Model Parameters: {param_count:,}")
        if hasattr(model, 'calculate_receptive_field'):
            rf = model.calculate_receptive_field()
            print(f"📏 Receptive Field: {rf}")
    
    # Training results
    best_train_acc = max(history['train_accuracies'])
    best_test_acc = max(history['test_accuracies'])
    final_train_acc = history['train_accuracies'][-1]
    final_test_acc = history['test_accuracies'][-1]
    
    print(f"\n📈 Training Results:")
    print(f"  Best Train Accuracy: {best_train_acc:.2f}%")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"  Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"  Total Epochs: {len(history['epochs'])}")
    
    # Overfitting analysis
    overfitting = final_train_acc - final_test_acc
    print(f"\n🔍 Overfitting Analysis:")
    print(f"  Train-Test Gap: {overfitting:.2f}%")
    if overfitting > 5:
        print("  ⚠️  Potential overfitting detected")
    elif overfitting < 1:
        print("  ✅ Good generalization")
    else:
        print("  ✅ Reasonable generalization")
    
    # Configuration
    if config:
        print(f"\n⚙️  Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")


def create_directory_structure(base_dir, subdirs=None):
    """
    Create directory structure for training.
    
    Args:
        base_dir: Base directory path
        subdirs: List of subdirectories to create (optional)
    """
    if subdirs is None:
        subdirs = ['checkpoints', 'logs', 'plots', 'models']
    
    os.makedirs(base_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    print(f"📁 Directory structure created in: {base_dir}")


def setup_optimizer_and_scheduler(model, config, steps_per_epoch):
    """
    Setup optimizer and scheduler based on configuration.
    
    Args:
        model: Neural network model
        config: Training configuration
        steps_per_epoch: Number of training steps per epoch
        
    Returns:
        tuple: (optimizer, scheduler, scheduler_type)
    """
    import torch.optim as optim
    
    # Setup optimizer
    if config.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Setup scheduler
    scheduler_type = config.scheduler
    
    if config.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.step_size, 
            gamma=config.gamma
        )
    elif config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config.T_0, 
            T_mult=1,
            eta_min=config.eta_min
        )
    elif config.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5
        )
    elif config.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor
        )
        scheduler_type = 'cyclic'  # OneCycleLR updates per batch like cyclic
    elif config.scheduler == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.learning_rate / 10,
            max_lr=config.learning_rate * 10,
            step_size_up=steps_per_epoch * 4,
            mode='triangular2'
        )
        scheduler_type = 'cyclic'
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    return optimizer, scheduler, scheduler_type


def test_architecture_only(model_class, num_classes=100, dropout=0.1, input_shape=(3, 32, 32)):
    """
    Test only the model architecture without training.
    
    Args:
        model_class: Model class to test
        num_classes: Number of output classes
        dropout: Dropout rate
        input_shape: Input tensor shape
        
    Returns:
        model: Tested model instance
    """
    print("🔍 Testing Model Architecture")
    print("=" * 50)
    
    # Test model creation
    model = model_class(num_classes=num_classes, dropout=dropout)
    
    # Test with input
    x = torch.randn(1, *input_shape)
    output = model(x)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of classes: {output.shape[1]}")
    
    return model


def test_utils():
    """
    Test utility functions.
    """
    print("🔍 Testing Utility Functions")
    print("=" * 40)
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test random seed
    set_random_seed(42)
    
    # Test time formatting
    print(f"Time formatting: {format_time(3661)}")
    
    # Test accuracy calculation
    outputs = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    acc = calculate_accuracy(outputs, targets)
    print(f"Accuracy calculation: {acc:.2f}%")
    
    print("✅ Utility functions test completed!")


if __name__ == "__main__":
    test_utils()
