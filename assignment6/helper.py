"""
Helper utilities for MNIST training pipeline.
Contains utility functions for device detection, random seed setting, and other common operations.
"""

import torch
import random
import numpy as np
import os

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


def get_model_summary(model, input_size=(1, 28, 28)):
    """
    Get a simple model summary showing parameter count.
    
    Args:
        model: The neural network model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    model.to('cpu')
    
    if TORCHSUMMARY_AVAILABLE:
        summary(model, input_size=input_size)
    else:
        # Fallback: just show parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return total_params, trainable_params


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        filepath: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        filepath: Path to the checkpoint file
        
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
