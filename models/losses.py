"""
Loss Functions for Extreme Class Imbalance Mitigation
CSCI 2026 - Passive Acoustic Monitoring Benchmark

Implements loss functions evaluated across the 18 experimental trials:
1. Focal Loss (Lin et al., 2017) with gamma=2.0 (Winner)
2. Class-Weighted Cross-Entropy (Inverse frequency)
3. Effective Sample Weighting Loss (Cui et al., 2019)
4. Standard Cross-Entropy Baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for long-tailed classification.
    Formula: L_focal = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Class indices [batch_size] or target probabilities [batch_size, num_classes] (for Mixup)
        """
        num_classes = logits.size(-1)

        # Handle hard targets vs soft targets (Mixup / Label Smoothing)
        if targets.ndim == 1:
            if self.label_smoothing > 0.0:
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
                targets = smooth_targets
            else:
                targets = F.one_hot(targets, num_classes=num_classes).float()

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        
        loss = -targets * focal_weight * log_probs

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            loss = loss * self.alpha.unsqueeze(0)

        loss = torch.sum(loss, dim=-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


def compute_effective_sample_weights(
    class_counts: List[int],
    beta: float = 0.999
) -> torch.Tensor:
    """
    Computes class weights based on Effective Number of Samples (Cui et al., 2019).
    Formula: E_n = (1 - beta^n) / (1 - beta)
             weight_c = 1.0 / E_n (normalized)
    """
    class_counts = np.array(class_counts, dtype=np.float32)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / np.sum(weights) * len(class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_inverse_frequency_weights(
    class_counts: List[int],
    power: float = 1.0
) -> torch.Tensor:
    """
    Computes inverse frequency class weights.
    Formula: weight_c = (N / N_c)^power
    """
    counts = np.array(class_counts, dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    weights = np.power(np.sum(counts) / counts, power)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_loss_function(
    loss_type: str = "focal",
    class_counts: Optional[List[int]] = None,
    gamma: float = 2.0,
    beta: float = 0.999,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Factory function to retrieve requested loss function.

    Args:
        loss_type: 'focal', 'class_weighted', 'effective_samples', 'ce'
        class_counts: List of sample counts per class (required for weighted losses)
        gamma: Focal Loss focusing parameter (default: 2.0)
        beta: Effective samples hyperparameter (default: 0.999)
        label_smoothing: Label smoothing epsilon

    Returns:
        nn.Module: Configured loss criterion
    """
    loss_type = loss_type.lower()

    if loss_type == "focal":
        return FocalLoss(gamma=gamma, label_smoothing=label_smoothing)

    elif loss_type == "class_weighted":
        if class_counts is None:
            raise ValueError("class_counts must be provided for class_weighted loss.")
        weights = compute_inverse_frequency_weights(class_counts)
        return FocalLoss(gamma=0.0, alpha=weights, label_smoothing=label_smoothing)

    elif loss_type == "effective_samples":
        if class_counts is None:
            raise ValueError("class_counts must be provided for effective_samples loss.")
        weights = compute_effective_sample_weights(class_counts, beta=beta)
        return FocalLoss(gamma=0.0, alpha=weights, label_smoothing=label_smoothing)

    elif loss_type in ["ce", "cross_entropy"]:
        return FocalLoss(gamma=0.0, label_smoothing=label_smoothing)

    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'. Choose from ['focal', 'class_weighted', 'effective_samples', 'ce']")


if __name__ == "__main__":
    # Test forward passes and weight generation
    dummy_counts = [990, 500, 100, 20, 5] + [10] * 201  # Total 206 classes
    logits = torch.randn(4, 206)
    targets = torch.tensor([0, 2, 4, 205])

    for l_type in ["focal", "class_weighted", "effective_samples", "ce"]:
        criterion = get_loss_function(l_type, class_counts=dummy_counts, gamma=2.0)
        loss = criterion(logits, targets)
        print(f"Loss Type: {l_type:<20} | Loss Value: {loss.item():.4f}")
