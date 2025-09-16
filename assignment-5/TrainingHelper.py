"""
TrainingHelper.py

A utility class containing static methods for training and evaluating neural networks.
All methods are static for easy import and use without instantiation.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class TrainingHelper:
    """
    A helper class containing static methods for neural network training and evaluation.
    All methods are static, so they can be used directly without creating an instance.
    """
    
    @staticmethod
    def train_epoch(model, device, train_loader, optimizer, criterion, epoch=None):
        """
        Train the model for one epoch.
        
        Args:
            model: The neural network model
            device: Device to run training on (cuda, mps, cpu)
            train_loader: DataLoader for training data
            optimizer: Optimizer for updating model parameters
            criterion: Loss function
            epoch: Current epoch number (optional, for display purposes)
            
        Returns:
            tuple: (average_loss, accuracy) for the epoch
        """
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}' if epoch else 'Training', leave=False)
        
        for data, target in progress_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            current_accuracy = 100. * correct_predictions / total_samples
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_accuracy:.2f}%'})
        
        avg_loss = total_loss / total_samples
        accuracy = 100. * correct_predictions / total_samples
        #print(f'  Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%')
        return avg_loss, accuracy

    @staticmethod
    def evaluate_model(model, device, test_loader, criterion, dataset_name="Test"):
        """
        Evaluate the model on a dataset.
        
        Args:
            model: The neural network model
            device: Device to run evaluation on (cuda, mps, cpu)
            test_loader: DataLoader for test data
            criterion: Loss function
            dataset_name: Name of the dataset for display purposes
            
        Returns:
            tuple: (average_loss, accuracy) for the evaluation
        """
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(test_loader, desc=f'{dataset_name} Evaluation', leave=False)
        
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
        #print(f'  {dataset_name} Loss: {avg_loss:.4f}, {dataset_name} Accuracy: {accuracy:.2f}%')
        return avg_loss, accuracy

    @staticmethod
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

    @staticmethod
    def set_seed(seed=42):
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed value (default: 42)
        """
        import random
        import numpy as np
        import os
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"🌱 Random seed set to {seed}")

    @staticmethod
    def get_model_summary(model):
        """
        Get a simple model summary showing parameter count.
        
        Args:
            model: The neural network model
            
        Returns:
            tuple: (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🏗️  Model Parameters: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        print(f"   Non-trainable: {total_params - trainable_params:,}")
        
        return total_params, trainable_params

    @staticmethod
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
        #print(f"💾 Checkpoint saved to {filepath}")

    @staticmethod
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
        #print(f"📂 Checkpoint loaded from {filepath}")
        return checkpoint

    @staticmethod
    def setup_training_config(num_epochs=10, early_stopping_patience=5, min_delta=0.001, 
                             save_best_only=True, checkpoint_dir='./checkpoints'):
        """
        Setup training configuration and initialize training history.
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            save_best_only: Whether to save only the best model
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            tuple: (config_dict, training_history, checkpoint_paths)
        """
        import os
        
        # Training configuration
        config = {
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'min_delta': min_delta,
            'save_best_only': save_best_only
        }
        
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
        
        return config, training_history, checkpoint_paths
