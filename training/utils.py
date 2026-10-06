import os
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int = 42) -> None:
    """
    Sets random seed across python, numpy, and PyTorch for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training epochs.
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def apply_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies Mixup data augmentation to input spectrogram tensors and target labels.

    Args:
        x: Input tensor batch [B, C, H, W]
        y: Target label tensor [B] or [B, C_classes]
        alpha: Beta distribution parameter (default: 0.4)

    Returns:
        mixed_x: Interpolated input batch
        y_a: Targets for first component
        y_b: Targets for second component
        lam: Lambda mixing ratio
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def apply_specaugment(
    spec: torch.Tensor,
    freq_mask_max: int = 15,
    time_mask_max: int = 20,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
) -> torch.Tensor:
    """
    Applies SpecAugment (Frequency and Time Masking) to 2D Mel-spectrograms.

    Args:
        spec: Spectrogram tensor [C, F, T] or [B, C, F, T]
        freq_mask_max: Maximum frequency channels to mask
        time_mask_max: Maximum time steps to mask
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply

    Returns:
        Augmented spectrogram tensor
    """
    augmented = spec.clone()
    is_batched = augmented.ndim == 4

    if not is_batched:
        augmented = augmented.unsqueeze(0)

    _, _, num_freqs, num_frames = augmented.shape

    for b in range(augmented.size(0)):
        # Frequency masking
        for _ in range(num_freq_masks):
            f_len = random.randint(0, freq_mask_max)
            f_start = random.randint(0, max(1, num_freqs - f_len))
            augmented[b, :, f_start : f_start + f_len, :] = 0.0

        # Time masking
        for _ in range(num_time_masks):
            t_len = random.randint(0, time_mask_max)
            t_start = random.randint(0, max(1, num_frames - t_len))
            augmented[b, :, :, t_start : t_start + t_len] = 0.0

    return augmented.squeeze(0) if not is_batched else augmented


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculates primary performance evaluation metrics:
    Top-1 Accuracy, Macro-F1, and Weighted-F1.

    Args:
        y_true: Ground truth target array [N]
        y_pred: Predicted class label array [N]

    Returns:
        Dictionary containing calculated metrics
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def save_checkpoint(state: dict, is_best: bool, save_dir: str = "checkpoints", filename: str = "checkpoint.pth") -> None:
    """
    Saves model state and metadata checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_filepath)
        print(f" Saved new best model checkpoint to: {best_filepath}")
