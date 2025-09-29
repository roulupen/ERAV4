#!/usr/bin/env python3
"""
Simple test script to verify the training pipeline works.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import create_cifar10_model
from data import get_cifar10_data_loaders
from trainer import create_trainer
from utils import set_random_seed, get_device

def test_training():
    """Test the training pipeline with a few epochs."""
    print("🧪 Testing CIFAR-10 Training Pipeline")
    print("=" * 50)
    
    # Set random seed
    set_random_seed(42)
    
    # Get device
    device = get_device()
    
    # Load data with small batch size for testing
    print("📥 Loading data...")
    train_loader, test_loader = get_cifar10_data_loaders(
        batch_size=32,
        num_workers=0,  # Use 0 for testing
        augment=True
    )
    
    # Create model
    print("🏗️ Creating model...")
    model = create_cifar10_model()
    model = model.to(device)
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.NLLLoss()
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion
    )
    
    # Test single epoch
    print("🚀 Testing single epoch...")
    train_loss, train_acc = trainer.train_epoch(1)
    test_loss, test_acc = trainer.test_epoch()
    
    print(f"✅ Training test completed!")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return True

if __name__ == "__main__":
    test_training()
