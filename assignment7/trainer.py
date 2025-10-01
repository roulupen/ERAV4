"""
Generic Training Module
Contains reusable training functions that work with any dataset and model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from typing import Dict, Any, Optional, Tuple, Callable
from tqdm import tqdm

# Import utils functions locally to avoid circular imports
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath, scheduler=None, additional_info=None):
    """Save model checkpoint with comprehensive information."""
    import torch
    import time
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
    """Load model checkpoint."""
    import torch
    if device is None:
        device = torch.device('cpu')
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint

def calculate_accuracy(outputs, targets):
    """Calculate accuracy given predictions and targets."""
    import torch
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100.0 * correct / total

def format_time(seconds):
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def create_directory_structure(base_dir, subdirs=None):
    """Create directory structure for training."""
    import os
    if subdirs is None:
        subdirs = ['checkpoints', 'logs', 'plots', 'models']
    os.makedirs(base_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    print(f"📁 Directory structure created in: {base_dir}")


class Trainer:
    """
    Generic trainer class for neural network training.
    """
    
    def __init__(self, model, device, train_loader, test_loader, 
                 optimizer, criterion, scheduler=None, config=None):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            device: Device to run training on
            train_loader: Training data loader
            test_loader: Test data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler (optional)
            config: Training configuration (optional)
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config or {}
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = {
            'train_losses': [],
            'train_accuracies': [],
            'test_losses': [],
            'test_accuracies': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Move model to device
        self.model.to(self.device)
        
        # Verify model is on correct device
        if hasattr(self.model, 'parameters'):
            model_device = next(self.model.parameters()).device
            print(f"🔧 Model moved to device: {model_device}")
            # Check if devices are equivalent (accounting for mps vs mps:0)
            if str(model_device) != str(self.device) and model_device.type != self.device.type:
                print(f"⚠️  Warning: Model device ({model_device}) != trainer device ({self.device})")
    
    def train_epoch(self, epoch_num: int, scheduler_type: str = 'step') -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch_num: Current epoch number
            scheduler_type: Type of scheduler ('step', 'plateau', 'cyclic')
            
        Returns:
            tuple: (average_loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch_num}', leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Update cyclic scheduler after each batch
            if self.scheduler is not None and scheduler_type == 'cyclic':
                self.scheduler.step()
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            current_accuracy = 100. * correct_predictions / total_samples
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_accuracy:.2f}%'})
        
        avg_loss = total_loss / total_samples
        accuracy = 100. * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def test_epoch(self) -> Tuple[float, float]:
        """
        Test the model for one epoch.
        
        Returns:
            tuple: (average_loss, accuracy) for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.test_loader, desc='Test Evaluation', leave=False)
        
        with torch.no_grad():
            for data, target in progress_bar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)
                
                current_accuracy = 100. * correct_predictions / total_samples
                progress_bar.set_postfix({'Acc': f'{current_accuracy:.2f}%'})
        
        avg_loss = total_loss / total_samples
        accuracy = 100. * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int = 20, early_stopping_patience: int = 5, 
              min_delta: float = 0.001, checkpoint_dir: str = './checkpoints',
              scheduler_type: str = 'step', save_best: bool = True,
              save_latest: bool = True, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the model with comprehensive tracking and checkpointing.
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            checkpoint_dir: Directory to save checkpoints
            scheduler_type: Type of scheduler ('step', 'plateau', 'cyclic')
            save_best: Whether to save best model
            save_latest: Whether to save latest model
            verbose: Whether to print training progress
            
        Returns:
            dict: Training history
        """
        # Setup checkpoint directory
        if save_best or save_latest:
            create_directory_structure(checkpoint_dir)
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        
        # Training variables
        best_accuracy = 0.0
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()
        
        if verbose:
            print(f"🚀 Starting training for {num_epochs} epochs...")
            print(f"⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 Checkpoint directory: {checkpoint_dir}")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch + 1
            
            # Training phase
            train_loss, train_accuracy = self.train_epoch(epoch + 1, scheduler_type)
            
            # Evaluation phase
            test_loss, test_accuracy = self.test_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if scheduler_type == 'plateau':
                    self.scheduler.step(test_accuracy)
                elif scheduler_type == 'cyclic':
                    # Cyclic scheduler is updated per batch, not per epoch
                    pass
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record training history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['train_accuracies'].append(train_accuracy)
            self.training_history['test_losses'].append(test_loss)
            self.training_history['test_accuracies'].append(test_accuracy)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['epochs'].append(epoch + 1)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            if verbose:
                # Print epoch summary
                print(f"Epoch {epoch + 1:2d}/{num_epochs} | "
                      f"Train: {train_loss:.4f} ({train_accuracy:.2f}%) | "
                      f"Test: {test_loss:.4f} ({test_accuracy:.2f}%) | "
                      f"LR: {current_lr:.6f} | Time: {format_time(epoch_time)}")
            
            # Check for best model
            if test_accuracy > best_accuracy + min_delta:
                best_accuracy = test_accuracy
                best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best:
                    save_checkpoint(
                        self.model, self.optimizer, epoch + 1, 
                        test_loss, test_accuracy, best_model_path,
                        scheduler=self.scheduler, additional_info=self.config
                    )
                    if verbose:
                        print(f"  🏆 New best model saved! (Test Acc: {test_accuracy:.2f}%)")
            else:
                patience_counter += 1
                if verbose:
                    print(f"  ⏳ No improvement ({patience_counter}/{early_stopping_patience})")
            
            # Save latest checkpoint
            if save_latest:
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, 
                    test_loss, test_accuracy, latest_checkpoint_path,
                    scheduler=self.scheduler, additional_info=self.config
                )
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                    print(f"   Best test accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
                break
        
        # Training completed
        total_time = time.time() - start_time
        if verbose:
            print(f"\n✅ Training completed!")
            print(f"   Total time: {format_time(total_time)}")
            print(f"   Best test accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
        
        # Update best accuracy
        self.best_accuracy = best_accuracy
        
        return self.training_history
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on the test set.
        
        Returns:
            tuple: (average_loss, accuracy) for the evaluation
        """
        return self.test_epoch()
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True, 
                       load_scheduler: bool = True) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            dict: Checkpoint data
        """
        checkpoint = load_checkpoint(
            self.model, self.optimizer, checkpoint_path,
            scheduler=self.scheduler if load_scheduler else None,
            device=self.device
        )
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary information.
        
        Returns:
            dict: Training summary
        """
        if not self.training_history['epochs']:
            return {}
        
        return {
            'total_epochs': len(self.training_history['epochs']),
            'best_train_accuracy': max(self.training_history['train_accuracies']),
            'best_test_accuracy': max(self.training_history['test_accuracies']),
            'final_train_accuracy': self.training_history['train_accuracies'][-1],
            'final_test_accuracy': self.training_history['test_accuracies'][-1],
            'best_epoch': self.training_history['test_accuracies'].index(max(self.training_history['test_accuracies'])) + 1,
            'overfitting_gap': self.training_history['train_accuracies'][-1] - self.training_history['test_accuracies'][-1]
        }


def create_trainer(model, device, train_loader, test_loader, optimizer, criterion, 
                  scheduler=None, config=None) -> Trainer:
    """
    Create a trainer instance.
    
    Args:
        model: Neural network model
        device: Device to run training on
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler (optional)
        config: Training configuration (optional)
        
    Returns:
        Trainer: Trainer instance
    """
    return Trainer(model, device, train_loader, test_loader, optimizer, criterion, scheduler, config)


def train_model(model, device, train_loader, test_loader, optimizer, criterion, scheduler=None,
                num_epochs=20, early_stopping_patience=5, min_delta=0.001,
                checkpoint_dir='./checkpoints', scheduler_type='step', config=None) -> Dict[str, Any]:
    """
    Convenience function to train a model.
    
    Args:
        model: Neural network model
        device: Device to run training on
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        early_stopping_patience: Patience for early stopping
        min_delta: Minimum improvement for early stopping
        checkpoint_dir: Directory to save checkpoints
        scheduler_type: Type of scheduler
        config: Training configuration (optional)
        
    Returns:
        dict: Training history
    """
    trainer = create_trainer(model, device, train_loader, test_loader, 
                           optimizer, criterion, scheduler, config)
    
    return trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
        checkpoint_dir=checkpoint_dir,
        scheduler_type=scheduler_type
    )


if __name__ == "__main__":
    print("🔍 Trainer module loaded successfully!")
    print("This module provides generic training functionality for any PyTorch model.")
