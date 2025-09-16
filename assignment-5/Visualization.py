"""
Visualization.py

A utility class containing static methods for visualizing training progress and model performance.
All methods are static for easy import and use without instantiation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Visualization:
    """
    A helper class containing static methods for neural network visualization.
    All methods are static, so they can be used directly without creating an instance.
    """
    
    @staticmethod
    def plot_training_history(training_history, figsize=(15, 10), save_path=None):
        """
        Plot comprehensive training history including loss and accuracy curves.
        
        Args:
            training_history: Dictionary containing training metrics
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        # Extract data from training history
        epochs = training_history['epochs']
        train_losses = training_history['train_losses']
        train_accuracies = training_history['train_accuracies']
        test_losses = training_history['test_losses']
        test_accuracies = training_history['test_accuracies']
        
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Progress Over Epochs', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss
        axs[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
        axs[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()
        
        # Plot 2: Training Accuracy
        axs[1, 0].plot(epochs, train_accuracies, 'g-', linewidth=2, marker='s', markersize=4, label='Training Accuracy')
        axs[1, 0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy (%)')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()
        
        # Plot 3: Test Loss
        axs[0, 1].plot(epochs, test_losses, 'r-', linewidth=2, marker='^', markersize=4, label='Test Loss')
        axs[0, 1].set_title('Test Loss', fontsize=14, fontweight='bold')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()
        
        # Plot 4: Test Accuracy
        axs[1, 1].plot(epochs, test_accuracies, 'm-', linewidth=2, marker='d', markersize=4, label='Test Accuracy')
        axs[1, 1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy (%)')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()
        
        # Add performance summary
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        final_test_acc = test_accuracies[-1] if test_accuracies else 0
        best_test_acc = max(test_accuracies) if test_accuracies else 0
        
        summary_text = f"""
        Performance Summary:
        • Final Training Accuracy: {final_train_acc:.2f}%
        • Final Test Accuracy: {final_test_acc:.2f}%
        • Best Test Accuracy: {best_test_acc:.2f}%
        • Total Epochs: {len(epochs)}
        """
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to {save_path}")
        
        plt.show()

    @staticmethod
    def plot_combined_metrics(training_history, figsize=(12, 5), save_path=None):
        """
        Plot combined loss and accuracy curves for better comparison.
        
        Args:
            training_history: Dictionary containing training metrics
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        epochs = training_history['epochs']
        train_losses = training_history['train_losses']
        train_accuracies = training_history['train_accuracies']
        test_losses = training_history['test_losses']
        test_accuracies = training_history['test_accuracies']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
        
        # Combined Loss Plot
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
        ax1.plot(epochs, test_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Test Loss')
        ax1.set_title('Loss Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Combined Accuracy Plot
        ax2.plot(epochs, train_accuracies, 'g-', linewidth=2, marker='o', markersize=4, label='Training Accuracy')
        ax2.plot(epochs, test_accuracies, 'm-', linewidth=2, marker='s', markersize=4, label='Test Accuracy')
        ax2.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Combined metrics plot saved to {save_path}")
        
        plt.show()

    @staticmethod
    def plot_learning_rate_schedule(training_history, figsize=(10, 6), save_path=None):
        """
        Plot learning rate schedule over training epochs.
        
        Args:
            training_history: Dictionary containing training metrics
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        epochs = training_history['epochs']
        learning_rates = training_history['learning_rates']
        
        plt.figure(figsize=figsize)
        plt.plot(epochs, learning_rates, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Learning rate schedule plot saved to {save_path}")
        
        plt.show()

    @staticmethod
    def visualize_samples(data_loader, num_samples=12, figsize=(12, 8)):
        """
        Visualize sample images from the dataset.
        
        Args:
            data_loader: DataLoader containing the dataset
            num_samples: Number of samples to display
            figsize: Figure size tuple (width, height)
        """
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle('MNIST Sample Images', fontsize=16)
        
        for i in range(min(num_samples, len(images))):
            row = i // 4
            col = i % 4
            
            # Denormalize image for display
            img = images[i].squeeze()
            img = img * 0.3081 + 0.1307  # Reverse normalization
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'Label: {labels[i].item()}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), save_path=None):
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names or range(len(cm)),
                   yticklabels=class_names or range(len(cm)))
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix plot saved to {save_path}")
        
        plt.show()

    @staticmethod
    def plot_model_predictions(model, data_loader, device, num_samples=12, figsize=(15, 10), save_path=None):
        """
        Plot model predictions on sample images.
        
        Args:
            model: Trained model
            data_loader: DataLoader containing test data
            device: Device to run inference on
            num_samples: Number of samples to display
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        model.eval()
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # Get predictions
        with torch.no_grad():
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            _, predicted = torch.max(outputs, 1)
        
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle('Model Predictions', fontsize=16)
        
        for i in range(min(num_samples, len(images))):
            row = i // 4
            col = i % 4
            
            # Denormalize image for display
            img = images[i].squeeze()
            img = img * 0.3081 + 0.1307  # Reverse normalization
            
            axes[row, col].imshow(img, cmap='gray')
            
            # Color code: green for correct, red for incorrect
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            color = 'green' if true_label == pred_label else 'red'
            
            axes[row, col].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Model predictions plot saved to {save_path}")
        
        plt.show()

    @staticmethod
    def plot_loss_curves(training_history, figsize=(12, 5), save_path=None):
        """
        Plot only loss curves (training and test).
        
        Args:
            training_history: Dictionary containing training metrics
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        epochs = training_history['epochs']
        train_losses = training_history['train_losses']
        test_losses = training_history['test_losses']
        
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
        plt.plot(epochs, test_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Test Loss')
        plt.title('Loss Curves', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Loss curves plot saved to {save_path}")
        
        plt.show()

    @staticmethod
    def plot_accuracy_curves(training_history, figsize=(12, 5), save_path=None):
        """
        Plot only accuracy curves (training and test).
        
        Args:
            training_history: Dictionary containing training metrics
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
        """
        epochs = training_history['epochs']
        train_accuracies = training_history['train_accuracies']
        test_accuracies = training_history['test_accuracies']
        
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_accuracies, 'g-', linewidth=2, marker='o', markersize=4, label='Training Accuracy')
        plt.plot(epochs, test_accuracies, 'm-', linewidth=2, marker='s', markersize=4, label='Test Accuracy')
        plt.title('Accuracy Curves', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Accuracy curves plot saved to {save_path}")
        
        plt.show()
