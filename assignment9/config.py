"""
Configuration Management Module for ImageNet Training
Handles training configuration, argument parsing, and hyperparameter management.
"""

import argparse
import json
import os
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """
    Training configuration dataclass for ImageNet.
    """
    # Model parameters
    model_name: str = 'resnet50_imagenet'
    num_classes: int = 1000
    dropout: float = 0.0
    pretrained: bool = False
    
    # Data parameters
    batch_size: int = 128
    num_workers: int = 4
    data_dir: str = '/kaggle/input/imagenet-object-localization-challenge'
    augment: bool = True
    pin_memory: bool = True
    limit_samples: int = None  # Limit samples for testing
    
    # Training parameters
    epochs: int = 90
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    early_stopping_patience: int = 10
    min_delta: float = 0.001
    
    # Optimizer parameters
    optimizer: str = 'sgd'  # 'adam', 'adamw', 'sgd'
    
    # Scheduler parameters
    scheduler: str = 'step'  # 'step', 'cosine', 'plateau', 'onecycle'
    step_size: int = 30  # For step scheduler
    gamma: float = 0.1  # For step scheduler
    T_0: int = 10  # For cosine scheduler
    eta_min: float = 1e-6  # For cosine scheduler
    
    # OneCycleLR scheduler parameters
    max_lr: float = 0.1
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    
    # Output parameters
    checkpoint_dir: str = './checkpoints'
    save_plots: bool = True
    save_model: bool = True
    
    # System parameters
    seed: int = 42
    verbose: bool = True
    
    # Target metrics
    target_accuracy: float = 70.0  # Top-1 accuracy target
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def get_default_config() -> TrainingConfig:
    """
    Get default training configuration.
    
    Returns:
        TrainingConfig: Default configuration
    """
    return TrainingConfig()


def print_config(config: TrainingConfig):
    """
    Print configuration in a formatted way.
    
    Args:
        config: Training configuration
    """
    print("⚙️  ImageNet Training Configuration")
    print("=" * 50)
    
    # Model parameters
    print("🏗️  Model:")
    print(f"  Name: {config.model_name}")
    print(f"  Classes: {config.num_classes}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Pretrained: {'✅' if config.pretrained else '❌'}")
    
    # Data parameters
    print("\n📊 Data:")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Data Dir: {config.data_dir}")
    print(f"  Augmentation: {'✅' if config.augment else '❌'}")
    print(f"  Pin Memory: {'✅' if config.pin_memory else '❌'}")
    if config.limit_samples:
        print(f"  ⚠️  Limited Samples: {config.limit_samples}")
    
    # Training parameters
    print("\n🚀 Training:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Momentum: {config.momentum}")
    print(f"  Early Stopping: {config.early_stopping_patience} epochs")
    
    # Optimizer and Scheduler
    print("\n🔧 Optimizer & Scheduler:")
    print(f"  Optimizer: {config.optimizer.upper()}")
    print(f"  Scheduler: {config.scheduler}")
    
    # Output parameters
    print("\n💾 Output:")
    print(f"  Checkpoint Dir: {config.checkpoint_dir}")
    print(f"  Save Plots: {'✅' if config.save_plots else '❌'}")
    print(f"  Save Model: {'✅' if config.save_model else '❌'}")
    
    # Target
    print("\n🎯 Target:")
    print(f"  Accuracy: {config.target_accuracy}%")
    
    print("=" * 50)


if __name__ == "__main__":
    config = get_default_config()
    print_config(config)

