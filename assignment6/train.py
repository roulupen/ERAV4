"""
Training script for MNIST classification using MNISTNet4Block model.
This script provides a comprehensive training pipeline with data loading,
model setup, training loop, and evaluation.

Refactored Functions:
- train_epoch(): Train model for one epoch
- test_epoch(): Test model for one epoch  
- train_model(): Complete training pipeline using the above functions
- evaluate_model(): Evaluate model on test set
- train_single_epoch(): Alias for train_epoch for clarity

Note: Model, optimizer, criterion, and scheduler should be set up separately.

Usage Examples:
    # Setup model and optimizer (do this yourself)
    model = MNISTNet4Block(dropout_prob=0.05).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    criterion = nn.NLLLoss()
    
    # Complete training pipeline
    training_history = train_model(model, device, train_loader, test_loader, optimizer, criterion, scheduler)
    
    # Train single epoch
    train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch_num)
    
    # Test single epoch
    test_loss, test_acc = test_epoch(model, device, test_loader, criterion)
    
    # Evaluate model
    test_loss, test_acc = evaluate_model(model, device, test_loader, criterion)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse
import sys
from tqdm import tqdm

# Import the model and helper functions
from model import (
    MNISTNet4BlockWithBatchNormMaxPoolConvFinal,
    MNISTNet4BlockWithBatchNormMaxPoolDropoutConvFinal,
    MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling
)
from helper import set_random_seed, get_device, get_model_summary, save_checkpoint
from data import get_mnist_data_loaders, get_cutout_data_loaders, plot_training_history


def train_epoch(model, device, train_loader, optimizer, criterion, epoch_num, scheduler=None, scheduler_type='step'):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        device: Device to run training on
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        epoch_num: Current epoch number
        
    Returns:
        tuple: (average_loss, accuracy) for the epoch
    """
    model.to(device)
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch_num}', leave=False)
    
    for data, target in progress_bar:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update cyclic scheduler after each batch
        if scheduler is not None and scheduler_type == 'cyclic':
            scheduler.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
        
        current_accuracy = 100. * correct_predictions / total_samples
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_accuracy:.2f}%'})
    
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct_predictions / total_samples
    
    return avg_loss, accuracy


def test_epoch(model, device, test_loader, criterion):
    """
    Test the model for one epoch.
    
    Args:
        model: The neural network model
        device: Device to run testing on
        test_loader: DataLoader for test data
        criterion: Loss function
        
    Returns:
        tuple: (average_loss, accuracy) for the epoch
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(test_loader, desc='Test Evaluation', leave=False)
    
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            current_accuracy = 100. * correct_predictions / total_samples
            progress_bar.set_postfix({'Acc': f'{current_accuracy:.2f}%'})
    
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct_predictions / total_samples
    
    return avg_loss, accuracy



def train_model(model, device, train_loader, test_loader, optimizer, criterion, scheduler,
                num_epochs=20, early_stopping_patience=5, min_delta=0.001,
                checkpoint_dir='./checkpoints', scheduler_type='step'):
    """
    Train the model with comprehensive tracking and checkpointing.
    
    Args:
        model: The neural network model
        device: Device to run training on
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        early_stopping_patience: Patience for early stopping
        min_delta: Minimum improvement for early stopping
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        dict: Training history containing losses, accuracies, and other metrics
    """
    
    # Initialize training history
    training_history = {
        'train_losses': [],
        'train_accuracies': [],
        'test_losses': [],
        'test_accuracies': [],
        'learning_rates': [],
        'epochs': []
    }
    
    # Setup checkpoint directory and paths
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_mnist_model.pth')
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    checkpoint_paths = {
        'best_model': best_model_path,
        'latest_checkpoint': checkpoint_path,
        'checkpoint_dir': checkpoint_dir
    }
    
    print("📁 Training Setup Complete:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Best model path: {best_model_path}")
    print(f"  Latest checkpoint: {checkpoint_path}")
    
    # Training variables
    best_test_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase - using separate function
        train_loss, train_accuracy = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch + 1, scheduler, scheduler_type
        )
        
        # Evaluation phase - using separate function
        test_loss, test_accuracy = test_epoch(
            model, device, test_loader, criterion
        )
        
        # Update learning rate
        if scheduler_type == 'plateau':
            scheduler.step(test_accuracy)
        elif scheduler_type == 'cyclic':
            # Cyclic scheduler is updated per batch, not per epoch
            pass  # Already updated during training
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record training history
        training_history['train_losses'].append(train_loss)
        training_history['train_accuracies'].append(train_accuracy)
        training_history['test_losses'].append(test_loss)
        training_history['test_accuracies'].append(test_accuracy)
        training_history['learning_rates'].append(current_lr)
        training_history['epochs'].append(epoch + 1)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary in single line
        print(f"Epoch {epoch + 1:2d}/{num_epochs} | Train: {train_loss:.4f} ({train_accuracy:.2f}%) | Test: {test_loss:.4f} ({test_accuracy:.2f}%) | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Check for best model
        if test_accuracy > best_test_accuracy + min_delta:
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            save_checkpoint(model, optimizer, epoch + 1, test_loss, test_accuracy, checkpoint_paths['best_model'])
            print(f"  🏆 New best model saved! (Test Acc: {test_accuracy:.2f}%)")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, epoch + 1, test_loss, test_accuracy, checkpoint_paths['latest_checkpoint'])
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            print(f"   Best test accuracy: {best_test_accuracy:.2f}% (Epoch {best_epoch})")
            break
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"   Total time: {total_time/60:.2f} minutes")
    print(f"   Best test accuracy: {best_test_accuracy:.2f}% (Epoch {best_epoch})")
    
    return training_history


def evaluate_model(model, device, test_loader, criterion):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The trained neural network model
        device: Device to run evaluation on
        test_loader: DataLoader for test data
        criterion: Loss function
        
    Returns:
        tuple: (average_loss, accuracy) for the evaluation
    """
    return test_epoch(model, device, test_loader, criterion)


def train_single_epoch(model, device, train_loader, optimizer, criterion, epoch_num):
    """
    Train the model for a single epoch.
    
    Args:
        model: The neural network model
        device: Device to run training on
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        epoch_num: Current epoch number
        
    Returns:
        tuple: (average_loss, accuracy) for the epoch
    """
    return train_epoch(model, device, train_loader, optimizer, criterion, epoch_num)




def get_model_by_name(model_name, dropout_prob=0.1):
    """
    Get model instance by name.
    
    Args:
        model_name: Name of the model to create
        dropout_prob: Dropout probability
        
    Returns:
        Model instance
    """
    model_map = {
        'batchnorm': MNISTNet4BlockWithBatchNormMaxPoolConvFinal,
        'dropout': MNISTNet4BlockWithBatchNormMaxPoolDropoutConvFinal,
        'gap': MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling
    }
    
    if model_name not in model_map:
        available_models = ', '.join(model_map.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    
    model_class = model_map[model_name]
    return model_class(dropout_prob=dropout_prob)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='MNIST Classification Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  basic      - Basic MNISTNet4Block
  batchnorm  - With BatchNorm and MaxPool
  dropout    - With BatchNorm, MaxPool, and Dropout
  gap        - With BatchNorm, MaxPool, Dropout, and Global Average Pooling
  improved   - Improved model with GAP (recommended)
  improved_v2- Alternative improved model

Examples:
  python train.py --model improved --batch_size 32 --epochs 15
  python train.py --model gap --batch_size 64 --lr 0.001
  python train.py --model improved_v2 --batch_size 16 --epochs 20
  python train.py --model improved --use_cutout --cutout_intensity medium
  python train.py --model improved --cutout_prob 0.3 --epochs 20
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        type=str, 
        default='improved',
        help='Model to train (default: improved)'
    )
    
    # Data parameters
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training (default: 32)'
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
        help='Directory for MNIST data (default: ./data)'
    )
    
    # Cutout augmentation parameters
    parser.add_argument(
        '--cutout_prob', 
        type=float, 
        default=0.2,
        help='Cutout augmentation probability (0.0 to 1.0, default: 0.2)'
    )
    parser.add_argument(
        '--cutout_intensity', 
        type=str, 
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help='Cutout augmentation intensity (default: medium)'
    )
    parser.add_argument(
        '--use_cutout', 
        action='store_true',
        help='Enable Cutout augmentation'
    )
    
    # Model parameters
    parser.add_argument(
        '--dropout_prob', 
        type=float, 
        default=0.15,
        help='Dropout probability (default: 0.15)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=15,
        help='Number of training epochs (default: 15)'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001,
        help='Learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=1e-3,
        help='Weight decay (default: 1e-3)'
    )
    parser.add_argument(
        '--early_stopping_patience', 
        type=int, 
        default=8,
        help='Early stopping patience (default: 8)'
    )
    parser.add_argument(
        '--min_delta', 
        type=float, 
        default=0.001,
        help='Minimum improvement for early stopping (default: 0.001)'
    )
    
    # Optimizer selection
    parser.add_argument(
        '--optimizer', 
        type=str, 
        choices=['adam', 'adamw', 'sgd'],
        default='adamw',
        help='Optimizer to use (default: adamw)'
    )
    
    # Scheduler selection
    parser.add_argument(
        '--scheduler', 
        type=str, 
        choices=['step', 'cosine', 'plateau', 'cyclic'],
        default='cosine',
        help='Learning rate scheduler (default: cosine)'
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
        help='Save training plots'
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
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_optimizer_and_scheduler(model, args, train_loader):
    """
    Setup optimizer and scheduler based on arguments.
    
    Args:
        model: The neural network model
        args: Parsed command line arguments
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            momentum=0.9
        )
    
    # Setup scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=4, 
            gamma=0.1
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5, 
            T_mult=1,
            eta_min=1e-6
        )
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
    elif args.scheduler == 'cyclic':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-3 if args.batch_size >= 128 else 1.5e-3,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.15,
            div_factor=10.0,        # initial lr = max_lr/div_factor
            final_div_factor=1e3    # final lr ~ max_lr/1e3
        )
    
    return optimizer, scheduler


def main():
    """
    Main training function with command line argument support.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("🎯 MNIST Classification Training Pipeline")
    print("=" * 60)
    print(f"📋 Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer.upper()}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Dropout: {args.dropout_prob}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Cutout: {'✅' if args.use_cutout else '❌'}")
    if args.use_cutout:
        print(f"  Cutout Intensity: {args.cutout_intensity}")
        print(f"  Cutout Probability: {args.cutout_prob}")
    print(f"  Random Seed: {args.seed}")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load data
        print(f"\n📥 Loading MNIST dataset...")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Workers: {args.num_workers}")
        print(f"  Data directory: {args.data_dir}")
        
        if args.use_cutout:
            print(f"  Using Cutout augmentation with {args.cutout_intensity} intensity")
            train_loader, test_loader = get_cutout_data_loaders(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                data_dir=args.data_dir,
                pin_memory=False,
                cutout_intensity=args.cutout_intensity
            )
        else:
            train_loader, test_loader = get_mnist_data_loaders(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                data_dir=args.data_dir,
                pin_memory=False,  # Default to False
                cutout_prob=args.cutout_prob
            )
        
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Setup model
        print(f"\n🏗️ Setting up {args.model} model...")
        model = get_model_by_name(args.model, args.dropout_prob)
        model = model.to(device)
        
        # Count parameters
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Total parameters: {total_params:,}")
            print(f"  Under 8k limit: {'✅' if total_params < 8000 else '❌'}")
            if total_params < 8000:
                print(f"  Parameter efficiency: {total_params/8000*100:.1f}% of limit")
        
        # Print model summary
        # if args.verbose:
        #     print("\n📊 Model Architecture:")
        #     get_model_summary(model)
        print("\n📊 Model Architecture:")
        get_model_summary(model)
        
        # Setup optimizer and scheduler
        print(f"\n⚙️  Setting up {args.optimizer.upper()} optimizer and {args.scheduler} scheduler...")
        optimizer, scheduler = setup_optimizer_and_scheduler(model, args, train_loader)
        
        # Setup loss function
        criterion = nn.NLLLoss()
        print(f"  Loss function: NLLLoss")
        
        # Train model
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        print(f"⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        training_history = train_model(
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            num_epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            checkpoint_dir=args.checkpoint_dir,
            scheduler_type=args.scheduler
        )
        
        # Generate plots if requested
        if args.save_plots:
            print("\n📊 Generating training plots...")
            plot_path = f'./training_history_{args.model}.png'
            plot_training_history(training_history, save_path=plot_path)
            print(f"  Plot saved to: {plot_path}")
        
        # Final summary
        best_test_acc = max(training_history['test_accuracies'])
        final_test_acc = training_history['test_accuracies'][-1]
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
        print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"  Target Achievement: {'✅' if best_test_acc >= 99.4 else '❌'}")
        print(f"  Total Epochs: {len(training_history['epochs'])}")
        
        # Save final model info
        model_info = {
            'model_name': args.model,
            'best_accuracy': best_test_acc,
            'final_accuracy': final_test_acc,
            'total_epochs': len(training_history['epochs']),
            'parameters': total_params if 'total_params' in locals() else 'Unknown',
            'config': vars(args)
        }
        
        import json
        info_path = f'./{args.model}_training_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"  Training info saved to: {info_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
