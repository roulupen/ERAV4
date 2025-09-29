"""
Main Training Script for CIFAR-10 Classification
Entry point for training CIFAR-10 models with the specified architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
from typing import Dict, Any

# Import our modules
from model import create_cifar10_model, test_model_architecture
from data import get_cifar10_data_loaders, get_dataset_info
from trainer import create_trainer, train_model
from utils import (
    set_random_seed, get_device, get_model_summary, 
    plot_training_history, save_training_info, print_training_summary
)
from config import parse_arguments, print_config, validate_config


def setup_optimizer_and_scheduler(model, config):
    """
    Setup optimizer and scheduler based on configuration.
    
    Args:
        model: Neural network model
        config: Training configuration
        
    Returns:
        tuple: (optimizer, scheduler)
    """
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
            momentum=config.momentum
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Setup scheduler
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
            patience=3
        )
    elif config.scheduler == 'cyclic':
        # Calculate steps per epoch (approximate)
        steps_per_epoch = 50000 // config.batch_size  # CIFAR-10 has 50k training samples
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate * 10,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=1e3
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    return optimizer, scheduler


def main():
    """
    Main training function.
    """
    print("🎯 CIFAR-10 Classification Training Pipeline")
    print("=" * 60)
    
    # Parse command line arguments
    config = parse_arguments()
    
    # Print configuration
    if config.verbose:
        print_config(config)
    
    # Validate configuration
    if not validate_config(config):
        print("❌ Invalid configuration. Exiting.")
        sys.exit(1)
    
    # Set random seed for reproducibility
    set_random_seed(config.seed)
    
    # Get device
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load data
        print(f"\n📥 Loading CIFAR-10 dataset...")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Workers: {config.num_workers}")
        print(f"  Data directory: {config.data_dir}")
        print(f"  Augmentation: {'✅' if config.augment else '❌'}")
        
        train_loader, test_loader = get_cifar10_data_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            data_dir=config.data_dir,
            augment=config.augment,
            pin_memory=None  # Auto-detect based on device
        )
        
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Setup model
        print(f"\n🏗️ Setting up {config.model_name} model...")
        model = create_cifar10_model(
            num_classes=config.num_classes, 
            use_fc=config.use_fc
        )
        model = model.to(device)
        
        # Verify model is on correct device
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")
        
        # Test model architecture
        print("\n🔍 Model Architecture Analysis:")
        test_model_architecture()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Under 200k limit: {'✅' if total_params < 200000 else '❌'}")
        if total_params < 200000:
            print(f"  Parameter efficiency: {total_params/200000*100:.1f}% of limit")
        
        # Calculate receptive field
        if hasattr(model, 'calculate_receptive_field'):
            rf = model.calculate_receptive_field()
            print(f"  Receptive field: {rf}")
            print(f"  RF > 44: {'✅' if rf > 44 else '❌'}")
        
        # Print model summary
        if config.verbose:
            print("\n📊 Model Architecture:")
            get_model_summary(model, input_size=(3, 32, 32))
        
        # Setup optimizer and scheduler
        print(f"\n⚙️  Setting up {config.optimizer.upper()} optimizer and {config.scheduler} scheduler...")
        optimizer, scheduler = setup_optimizer_and_scheduler(model, config)
        
        # Setup loss function
        criterion = nn.NLLLoss()
        print(f"  Loss function: NLLLoss")
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            config=config.to_dict()
        )
        
        # Train model
        print(f"\n🚀 Starting training for {config.epochs} epochs...")
        print(f"⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        training_history = trainer.train(
            num_epochs=config.epochs,
            early_stopping_patience=config.early_stopping_patience,
            min_delta=config.min_delta,
            checkpoint_dir=config.checkpoint_dir,
            scheduler_type=config.scheduler,
            verbose=config.verbose
        )
        
        # Generate plots if requested
        if config.save_plots:
            print("\n📊 Generating training plots...")
            plot_path = f'./training_history_{config.model_name}.png'
            plot_training_history(training_history, save_path=plot_path, show_plot=False)
            print(f"  Plot saved to: {plot_path}")
        
        # Print training summary
        print_training_summary(training_history, model, config.to_dict())
        
        # Final evaluation
        final_test_loss, final_test_acc = trainer.evaluate()
        
        # Check if target accuracy was achieved
        target_achieved = final_test_acc >= config.target_accuracy
        
        print(f"\n🎉 Training completed!")
        print(f"📊 Final Results:")
        print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"  Target Accuracy: {config.target_accuracy}%")
        print(f"  Target Achievement: {'✅' if target_achieved else '❌'}")
        
        # Save final model info
        model_info = {
            'model_name': config.model_name,
            'final_accuracy': final_test_acc,
            'best_accuracy': max(training_history['test_accuracies']),
            'total_epochs': len(training_history['epochs']),
            'parameters': total_params,
            'receptive_field': model.calculate_receptive_field() if hasattr(model, 'calculate_receptive_field') else 'Unknown',
            'target_achieved': target_achieved,
            'config': config.to_dict(),
            'training_history': training_history
        }
        
        info_path = f'./{config.model_name}_training_info.json'
        save_training_info(model_info, info_path)
        
        # Save model if requested
        if config.save_model:
            model_path = f'./{config.model_name}_final.pth'
            torch.save(model.state_dict(), model_path)
            print(f"  Model saved to: {model_path}")
        
        return model_info
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def test_architecture_only():
    """
    Test only the model architecture without training.
    """
    print("🔍 Testing CIFAR-10 Model Architecture")
    print("=" * 50)
    
    # Test model creation
    model = create_cifar10_model()
    
    # Test with CIFAR-10 input
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {model.get_parameter_count():,}")
    print(f"Receptive field: {model.calculate_receptive_field()}")
    
    # Check requirements
    print(f"\n📋 Requirements Check:")
    print(f"  Parameters < 200k: {'✅' if model.get_parameter_count() < 200000 else '❌'}")
    print(f"  RF > 44: {'✅' if model.calculate_receptive_field() > 44 else '❌'}")
    print(f"  Uses GAP: {'✅' if hasattr(model, 'gap') else '❌'}")
    print(f"  Uses Depthwise Separable: {'✅' if hasattr(model, 'conv2') and 'DepthwiseSeparableConv2d' in str(type(model.conv2)) else '❌'}")
    print(f"  Uses Dilated Conv: {'✅' if hasattr(model, 'conv3') and 'DilatedConv2d' in str(type(model.conv3)) else '❌'}")
    
    return model


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test-architecture':
        test_architecture_only()
    else:
        main()
