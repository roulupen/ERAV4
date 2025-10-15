"""
Main Training Script for ImageNet Classification with ResNet50
Entry point for training ImageNet models with ResNet50 architecture.

Usage:
    python main.py --epochs 90 --batch_size 128
    python main.py --test-architecture
"""

import torch
import torch.nn as nn
import os
import sys
import time
from typing import Dict, Any

# Import common modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.utils import (
    get_device, set_random_seed, get_model_summary,
    plot_training_history, save_training_info, print_training_summary,
    setup_optimizer_and_scheduler, test_architecture_only
)
from common.trainer import create_trainer

# Import assignment-specific modules
from assignment9.data import get_imagenet_data_loaders
from assignment9.model import resnet50
from assignment9.config import TrainingConfig, print_config


def main():
    """
    Main training function.
    """
    print("🎯 ImageNet Classification Training Pipeline with ResNet50")
    print("=" * 60)
    
    # Load configuration
    config = TrainingConfig()
    
    # Print configuration
    if config.verbose:
        print_config(config)
    
    # Set random seed for reproducibility
    set_random_seed(config.seed)
    
    # Get device
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load data
        train_loader, val_loader = get_imagenet_data_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            augment=config.augment,
            pin_memory=config.pin_memory,
            limit_samples=config.limit_samples
        )
        
        # Setup model
        print(f"\n🏗️  Setting up {config.model_name} model...")
        model = resnet50(
            num_classes=config.num_classes,
            dropout=config.dropout,
            pretrained=config.pretrained
        )
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n🔍 Model Architecture Analysis:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test model with sample input
        print(f"  Testing model with ImageNet input...")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(device)
            test_output = model(test_input)
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {test_output.shape}")
        model.train()
        
        # Setup optimizer and scheduler
        print(f"\n⚙️  Setting up {config.optimizer.upper()} optimizer and {config.scheduler} scheduler...")
        steps_per_epoch = len(train_loader)
        optimizer, scheduler, scheduler_type = setup_optimizer_and_scheduler(
            model, config, steps_per_epoch
        )
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        print(f"  Loss function: CrossEntropyLoss")
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            config=config.to_dict(),
            l2_lambda=0.0
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
        
        # Final evaluation
        final_test_loss, final_test_acc = trainer.evaluate()
        
        # Save results
        model_info = {
            'model_name': config.model_name,
            'dataset': 'ImageNet-1K',
            'num_classes': config.num_classes,
            'final_accuracy': final_test_acc,
            'parameters': total_params,
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
        
        print(f"\n🎉 Training completed!")
        print(f"📊 Final Test Accuracy: {final_test_acc:.2f}%")
        
        return model_info
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test-architecture':
        from assignment9.model import ResNet50ImageNet
        test_architecture_only(ResNet50ImageNet, num_classes=1000, dropout=0.0, input_shape=(3, 224, 224))
    else:
        main()

