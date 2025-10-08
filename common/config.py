"""
Configuration Management Module for CIFAR-100
Handles training configuration, argument parsing, and hyperparameter management.
"""

import argparse
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """
    Training configuration dataclass for CIFAR-100.
    """
    # Model parameters
    model_name: str = 'resnet18_cifar100'
    num_classes: int = 100
    dropout: float = 0.1
    
    # Data parameters
    batch_size: int = 128
    num_workers: int = 4
    data_dir: str = './data'
    augment: bool = True
    pin_memory: bool = True
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    l2_lambda: float = 1e-4  # L2 regularization strength
    early_stopping_patience: int = 15
    min_delta: float = 0.001
    
    # Optimizer parameters
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    momentum: float = 0.9  # For SGD
    
    # Scheduler parameters
    scheduler: str = 'cosine'  # 'step', 'cosine', 'plateau', 'cyclic', 'onecycle'
    step_size: int = 30  # For step scheduler
    gamma: float = 0.1  # For step scheduler
    T_0: int = 20  # For cosine scheduler
    eta_min: float = 1e-6  # For cosine scheduler
    
    # OneCycleLR scheduler parameters
    max_lr: float = 0.01  # Maximum learning rate for OneCycleLR
    pct_start: float = 0.3  # Percentage of cycle spent increasing lr
    div_factor: float = 25.0  # Initial lr = max_lr / div_factor
    final_div_factor: float = 1e4  # Final lr = initial_lr / final_div_factor
    
    # Output parameters
    checkpoint_dir: str = './checkpoints'
    save_plots: bool = True
    save_model: bool = True
    
    # System parameters
    seed: int = 42
    verbose: bool = True
    
    # Target metrics
    target_accuracy: float = 75.0  # CIFAR-100 is harder than CIFAR-10
    
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


def parse_arguments() -> TrainingConfig:
    """
    Parse command line arguments and return configuration.
    
    Returns:
        TrainingConfig: Parsed configuration
    """
    parser = argparse.ArgumentParser(
        description='CIFAR-100 Classification Training Pipeline with ResNet18',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --epochs 100 --batch_size 128 --lr 0.001
  python main.py --optimizer sgd --scheduler cosine --epochs 80
  python main.py --no_augment --batch_size 64
  python main.py --target_accuracy 75.0 --early_stopping_patience 20
        """
    )
    
    # Model parameters
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='resnet18_cifar100',
        help='Model name (default: resnet18_cifar100)'
    )
    parser.add_argument(
        '--num_classes', 
        type=int, 
        default=100,
        help='Number of output classes (default: 100)'
    )
    parser.add_argument(
        '--dropout', 
        type=float, 
        default=0.1,
        help='Dropout rate for all dropout layers (default: 0.1)'
    )
    
    # Data parameters
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=128,
        help='Batch size for training (default: 128)'
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of data loader workers (default: 4)'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./data',
        help='Directory for CIFAR-100 data (default: ./data)'
    )
    parser.add_argument(
        '--augment', 
        action='store_true',
        default=True,
        help='Enable data augmentation (default: True)'
    )
    parser.add_argument(
        '--no_augment', 
        action='store_false',
        dest='augment',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--pin_memory', 
        action='store_true',
        default=True,
        help='Pin memory for faster GPU transfer (default: True)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--lr', '--learning_rate',
        type=float, 
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=5e-4,
        help='Weight decay (default: 5e-4)'
    )
    parser.add_argument(
        '--l2_lambda', 
        type=float, 
        default=1e-4,
        help='L2 regularization strength (default: 1e-4)'
    )
    parser.add_argument(
        '--early_stopping_patience', 
        type=int, 
        default=15,
        help='Early stopping patience (default: 15)'
    )
    parser.add_argument(
        '--min_delta', 
        type=float, 
        default=0.001,
        help='Minimum improvement for early stopping (default: 0.001)'
    )
    
    # Optimizer parameters
    parser.add_argument(
        '--optimizer', 
        type=str, 
        choices=['adam', 'adamw', 'sgd'],
        default='adamw',
        help='Optimizer to use (default: adamw)'
    )
    parser.add_argument(
        '--momentum', 
        type=float, 
        default=0.9,
        help='Momentum for SGD (default: 0.9)'
    )
    
    # Scheduler parameters
    parser.add_argument(
        '--scheduler', 
        type=str, 
        choices=['step', 'cosine', 'plateau', 'cyclic', 'onecycle'],
        default='cosine',
        help='Learning rate scheduler (default: cosine)'
    )
    parser.add_argument(
        '--step_size', 
        type=int, 
        default=30,
        help='Step size for step scheduler (default: 30)'
    )
    parser.add_argument(
        '--gamma', 
        type=float, 
        default=0.1,
        help='Gamma for step scheduler (default: 0.1)'
    )
    parser.add_argument(
        '--T_0', 
        type=int, 
        default=20,
        help='T_0 for cosine scheduler (default: 20)'
    )
    parser.add_argument(
        '--eta_min', 
        type=float, 
        default=1e-6,
        help='Minimum learning rate (default: 1e-6)'
    )
    
    # OneCycleLR parameters
    parser.add_argument(
        '--max_lr', 
        type=float, 
        default=0.01,
        help='Maximum learning rate for OneCycleLR (default: 0.01)'
    )
    parser.add_argument(
        '--pct_start', 
        type=float, 
        default=0.3,
        help='Percentage of cycle spent increasing lr (default: 0.3)'
    )
    parser.add_argument(
        '--div_factor', 
        type=float, 
        default=25.0,
        help='Initial lr = max_lr / div_factor (default: 25.0)'
    )
    parser.add_argument(
        '--final_div_factor', 
        type=float, 
        default=1e4,
        help='Final lr = initial_lr / final_div_factor (default: 1e4)'
    )
    
    # Output parameters
    parser.add_argument(
        '--checkpoint_dir', 
        type=str, 
        default='./checkpoints',
        help='Directory to save checkpoints (default: ./checkpoints)'
    )
    parser.add_argument(
        '--save_plots', 
        action='store_true',
        default=True,
        help='Save training plots (default: True)'
    )
    parser.add_argument(
        '--no_plots', 
        action='store_false',
        dest='save_plots',
        help='Disable saving plots'
    )
    parser.add_argument(
        '--save_model', 
        action='store_true',
        default=True,
        help='Save trained model (default: True)'
    )
    
    # System parameters
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        default=True,
        help='Enable verbose logging (default: True)'
    )
    parser.add_argument(
        '--quiet', 
        action='store_false',
        dest='verbose',
        help='Disable verbose logging'
    )
    
    # Target metrics
    parser.add_argument(
        '--target_accuracy', 
        type=float, 
        default=75.0,
        help='Target accuracy to achieve (default: 75.0)'
    )
    
    # Config file
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file (JSON)'
    )
    
    args = parser.parse_args()
    
    # Load config from file if specified
    if args.config and os.path.exists(args.config):
        config = TrainingConfig.load(args.config)
        print(f"📁 Loaded configuration from: {args.config}")
    else:
        config = TrainingConfig()
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            setattr(config, key, value)
    
    return config


def get_default_config() -> TrainingConfig:
    """
    Get default training configuration.
    
    Returns:
        TrainingConfig: Default configuration
    """
    return TrainingConfig()


def create_config_from_dict(config_dict: Dict[str, Any]) -> TrainingConfig:
    """
    Create configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        TrainingConfig: Configuration object
    """
    return TrainingConfig(**config_dict)


def print_config(config: TrainingConfig):
    """
    Print configuration in a formatted way.
    
    Args:
        config: Training configuration
    """
    print("⚙️  CIFAR-100 Training Configuration")
    print("=" * 50)
    
    # Model parameters
    print("🏗️  Model:")
    print(f"  Name: {config.model_name}")
    print(f"  Classes: {config.num_classes}")
    print(f"  Dropout: {config.dropout}")
    
    # Data parameters
    print("\n📊 Data:")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Data Dir: {config.data_dir}")
    print(f"  Augmentation: {'✅' if config.augment else '❌'}")
    print(f"  Pin Memory: {'✅' if config.pin_memory else '❌'}")
    
    # Training parameters
    print("\n🚀 Training:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Early Stopping: {config.early_stopping_patience} epochs")
    print(f"  Min Delta: {config.min_delta}")
    
    # Optimizer parameters
    print("\n🔧 Optimizer:")
    print(f"  Type: {config.optimizer.upper()}")
    if config.optimizer == 'sgd':
        print(f"  Momentum: {config.momentum}")
    
    # Scheduler parameters
    print("\n📈 Scheduler:")
    print(f"  Type: {config.scheduler}")
    if config.scheduler == 'step':
        print(f"  Step Size: {config.step_size}")
        print(f"  Gamma: {config.gamma}")
    elif config.scheduler == 'cosine':
        print(f"  T_0: {config.T_0}")
        print(f"  Eta Min: {config.eta_min}")
    elif config.scheduler == 'onecycle':
        print(f"  Max LR: {config.max_lr}")
        print(f"  Pct Start: {config.pct_start}")
        print(f"  Div Factor: {config.div_factor}")
        print(f"  Final Div Factor: {config.final_div_factor}")
    
    # Output parameters
    print("\n💾 Output:")
    print(f"  Checkpoint Dir: {config.checkpoint_dir}")
    print(f"  Save Plots: {'✅' if config.save_plots else '❌'}")
    print(f"  Save Model: {'✅' if config.save_model else '❌'}")
    
    # System parameters
    print("\n🖥️  System:")
    print(f"  Seed: {config.seed}")
    print(f"  Verbose: {'✅' if config.verbose else '❌'}")
    
    # Target metrics
    print("\n🎯 Target:")
    print(f"  Accuracy: {config.target_accuracy}%")
    
    print("=" * 50)


def validate_config(config: TrainingConfig) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Training configuration
        
    Returns:
        bool: True if valid, False otherwise
    """
    errors = []
    
    # Validate positive values
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    if config.num_classes != 100:
        errors.append("Number of classes must be 100 for CIFAR-100")
    
    # Validate ranges
    if config.learning_rate > 1.0:
        errors.append("Learning rate seems too high (> 1.0)")
    
    if config.weight_decay < 0:
        errors.append("Weight decay must be non-negative")
    
    if config.early_stopping_patience < 1:
        errors.append("Early stopping patience must be at least 1")
    
    if config.target_accuracy < 0 or config.target_accuracy > 100:
        errors.append("Target accuracy must be between 0 and 100")
    
    if config.dropout < 0 or config.dropout > 1:
        errors.append("Dropout must be between 0 and 1")
    
    if errors:
        print("❌ Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print_config(config)
    
    if validate_config(config):
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors!")

