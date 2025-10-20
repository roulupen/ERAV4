"""
Mixup and CutMix Augmentation for Tiny ImageNet
These are advanced augmentation techniques that can boost accuracy by 3-5%
"""

import torch
import numpy as np


def mixup_data(x, y, alpha=1.0):
    """
    Mixup augmentation: blend two images and their labels.
    
    Args:
        x: Input images (batch)
        y: Input labels (batch)
        alpha: Mixup hyperparameter (default: 1.0)
    
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix.
    
    Args:
        size: Image size (B, C, H, W)
        lam: Lambda parameter
    
    Returns:
        Bounding box coordinates (x1, y1, x2, y2)
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation: cut and paste patches between images.
    
    Args:
        x: Input images (batch)
        y: Input labels (batch)
        alpha: CutMix hyperparameter (default: 1.0)
    
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def apply_mixup_cutmix(x, y, criterion, model, mixup_prob=0.5, cutmix_prob=0.5, 
                       mixup_alpha=1.0, cutmix_alpha=1.0):
    """
    Randomly apply Mixup or CutMix augmentation during training.
    
    Args:
        x: Input images
        y: Input labels
        criterion: Loss function
        model: Neural network model
        mixup_prob: Probability of applying Mixup (default: 0.5)
        cutmix_prob: Probability of applying CutMix (default: 0.5)
        mixup_alpha: Mixup alpha parameter (default: 1.0)
        cutmix_alpha: CutMix alpha parameter (default: 1.0)
    
    Returns:
        loss: Computed loss
        output: Model predictions
    """
    r = np.random.rand(1)
    
    if r < mixup_prob:
        # Apply Mixup
        x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
        output = model(x)
        loss = mixup_criterion(criterion, output, y_a, y_b, lam)
    elif r < mixup_prob + cutmix_prob:
        # Apply CutMix
        x, y_a, y_b, lam = cutmix_data(x, y, cutmix_alpha)
        output = model(x)
        loss = mixup_criterion(criterion, output, y_a, y_b, lam)
    else:
        # Normal training
        output = model(x)
        loss = criterion(output, y)
    
    return loss, output


# Example usage in training loop
"""
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    
    # Use Mixup/CutMix
    loss, output = apply_mixup_cutmix(
        data, target, criterion, model,
        mixup_prob=0.4,
        cutmix_prob=0.4,
        mixup_alpha=1.0,
        cutmix_alpha=1.0
    )
    
    loss.backward()
    optimizer.step()
"""

