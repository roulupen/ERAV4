"""
Main Training Script for CIFAR-100 Classification with ResNet18
Entry point for training CIFAR-100 models with ResNet architecture.

Usage:
    python main.py --dataset cifar100 --epochs 100
    python main.py --dataset cifar100 --test-architecture
"""

import torch
import torch.nn as nn
import os
import sys
import time
from typing import Dict, Any

# Import common modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common import (
    get_device, set_random_seed, get_model_summary,
    plot_training_history, save_training_info, print_training_summary,
    create_trainer, TrainingConfig, print_config, validate_config,
    setup_optimizer_and_scheduler, test_architecture_only
)

# Import assignment-specific modules
from assignment8.data import get_cifar100_data_loaders
from assignment8.model import ResNet18CIFAR100


def main():
    """
    Main training function.
    """
    print("🎯 CIFAR-100 Classification Training Pipeline with ResNet18")
    print("=" * 60)
    
    # Parse command line arguments using common config
    from common.config import parse_arguments
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
        print(f"\n📥 Loading CIFAR-100 dataset...")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Workers: {config.num_workers}")
        print(f"  Data directory: {config.data_dir}")
        print(f"  Augmentation: {'✅' if config.augment else '❌'}")
        
        train_loader, test_loader = get_cifar100_data_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            data_dir=config.data_dir,
            augment=config.augment,
            pin_memory=None  # Auto-detect based on device
        )
        
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Training samples: ~{len(train_loader) * config.batch_size:,}")
        print(f"  Test samples: ~{len(test_loader) * config.batch_size:,}")
        
        # Setup model
        print(f"\n🏗️  Setting up {config.model_name} model...")
        model = ResNet18CIFAR100(
            num_classes=config.num_classes, 
            dropout=config.dropout
        )
        model = model.to(device)
        
        # Verify model is on correct device
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n🔍 Model Architecture Analysis:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test model with sample input
        print(f"  Testing model with CIFAR-100 input...")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32).to(device)
            test_output = model(test_input)
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {test_output.shape}")
            print(f"  Output classes: {test_output.shape[1]}")
        model.train()
        
        # Print model summary
        if config.verbose:
            print("\n📊 Model Architecture:")
            get_model_summary(model, input_size=(3, 32, 32))
        
        # Setup optimizer and scheduler
        print(f"\n⚙️  Setting up {config.optimizer.upper()} optimizer and {config.scheduler} scheduler...")
        steps_per_epoch = len(train_loader)
        optimizer, scheduler, scheduler_type = setup_optimizer_and_scheduler(
            model, config, steps_per_epoch
        )
        
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Scheduler type: {scheduler_type}")
        
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
            config=config.to_dict(),
            l2_lambda=config.l2_lambda  # Add L2 regularization from config
        )
        
        # Train model
        print(f"\n🚀 Starting training for {config.epochs} epochs...")
        print(f"⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        training_history = trainer.train(
            num_epochs=config.epochs,
            early_stopping_patience=config.early_stopping_patience,
            min_delta=config.min_delta,
            checkpoint_dir=config.checkpoint_dir,
            scheduler_type=scheduler_type,
            verbose=config.verbose
        )
        
        # Generate plots if requested
        if config.save_plots:
            print("\n📊 Generating training plots...")
            plot_path = f'./training_history_{config.model_name}.png'
            plot_training_history(training_history, save_path=plot_path, show_plot=False)
            print(f"  Plot saved to: {plot_path}")
        
        # Print training summary
        if hasattr(training_history, 'get') and 'train_accuracies' in training_history:
            print_training_summary(training_history, model, config.to_dict())
        else:
            print("\n📊 Training completed successfully!")
            print(f"  Check training logs above for detailed results.")
        
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
        best_accuracy = final_test_acc
        total_epochs = config.epochs
        
        if hasattr(training_history, 'get') and 'test_accuracies' in training_history:
            best_accuracy = max(training_history['test_accuracies'])
            total_epochs = len(training_history.get('epochs', range(config.epochs)))
        
        model_info = {
            'model_name': config.model_name,
            'dataset': 'CIFAR-100',
            'num_classes': config.num_classes,
            'final_accuracy': final_test_acc,
            'best_accuracy': best_accuracy,
            'total_epochs': total_epochs,
            'parameters': total_params,
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




if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test-architecture':
        test_architecture_only(ResNet18CIFAR100, num_classes=100, dropout=0.1, input_shape=(3, 32, 32))
    else:
        main()

