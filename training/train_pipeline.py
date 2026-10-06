import os
import sys
import json
import argparse
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

# Add parent directories to sys.path to ensure modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_preprocessing.audio_preprocessing import BioacousticDataset
from models.backbones import get_model
from models.losses import get_loss_function
from training.utils import (
    set_seed,
    AverageMeter,
    apply_mixup,
    apply_specaugment,
    calculate_metrics,
    save_checkpoint,
)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
    use_specaugment: bool = True,
) -> float:
    """
    Executes one training epoch with optional Mixup and SpecAugment data augmentation.
    """
    model.train()
    loss_meter = AverageMeter("Train Loss", ":.4f")

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Apply SpecAugment on 2D Mel Spectrograms
        if use_specaugment:
            x_batch = apply_specaugment(x_batch, freq_mask_max=15, time_mask_max=20)

        # Apply Mixup regularization
        if use_mixup:
            x_batch, y_a, y_b, lam = apply_mixup(x_batch, y_batch, alpha=mixup_alpha)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
        else:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), n=x_batch.size(0))

    return loss_meter.avg


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """
    Evaluates the model on the validation dataset partition.
    """
    model.eval()
    loss_meter = AverageMeter("Val Loss", ":.4f")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            preds = torch.argmax(outputs, dim=1)

            loss_meter.update(loss.item(), n=x_batch.size(0))
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    return loss_meter.avg, metrics


def run_experiment(
    exp_id: str = "E6",
    config_path: str = "configs/preprocessing_config.json",
    matrix_path: str = "configs/experiments_matrix.json",
    train_manifest: str = "data/train_split.csv",
    val_manifest: str = "data/val_split.csv",
    output_dir: str = "results",
) -> None:
    """
    Main training execution function for a specific experiment in the 18-configuration matrix.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Experiment {exp_id} on Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load configuration parameters
    with open(config_path, "r") as f:
        audio_config = json.load(f)

    with open(matrix_path, "r") as f:
        exp_matrix = json.load(f)["experiments"]

    if exp_id not in exp_matrix:
        raise ValueError(f"Experiment ID {exp_id} not found in experiments_matrix.json")

    exp_info = exp_matrix[exp_id]
    print(f"Experiment Config: {exp_info['model']} | Loss: {exp_info['loss']} | Augmentation: {exp_info['augmentation']}")

    # Setup Datasets & DataLoaders
    train_dataset = BioacousticDataset(
        manifest_path=train_manifest,
        sample_rate=audio_config["audio_parameters"]["sample_rate"],
        duration=audio_config["audio_parameters"]["duration_seconds"],
        n_mels=audio_config["spectrogram_parameters"]["n_mels"],
        n_fft=audio_config["spectrogram_parameters"]["n_fft"],
        hop_length=audio_config["spectrogram_parameters"]["hop_length"],
        f_min=audio_config["spectrogram_parameters"]["f_min"],
        f_max=audio_config["spectrogram_parameters"]["f_max"],
        is_train=True,
    )

    val_dataset = BioacousticDataset(
        manifest_path=val_manifest,
        sample_rate=audio_config["audio_parameters"]["sample_rate"],
        duration=audio_config["audio_parameters"]["duration_seconds"],
        n_mels=audio_config["spectrogram_parameters"]["n_mels"],
        n_fft=audio_config["spectrogram_parameters"]["n_fft"],
        hop_length=audio_config["spectrogram_parameters"]["hop_length"],
        f_min=audio_config["spectrogram_parameters"]["f_min"],
        f_max=audio_config["spectrogram_parameters"]["f_max"],
        is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate Model, Loss, Optimizer, and Schedulers
    num_classes = audio_config["taxonomy_summary"]["total_species"]
    model = get_model(architecture=exp_info["model"], num_classes=num_classes, pretrained=True).to(device)

    criterion = get_loss_function(
        loss_type=exp_info["loss"],
        num_classes=num_classes,
        gamma=2.0,
        label_smoothing=0.1 if "LabelSmoothing" in exp_info["regularization"] else 0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Stochastic Weight Averaging (SWA) setup
    use_swa = "SWA" in exp_info["regularization"]
    if use_swa:
        swa_model = AveragedModel(model)
        swa_start_epoch = 20
        swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

    use_mixup = "Mixup" in exp_info["augmentation"]
    use_specaugment = "SpecAugment" in exp_info["augmentation"]

    best_val_f1 = 0.0
    history = []

    # Training Loop (30 Epochs)
    total_epochs = 30
    for epoch in range(1, total_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_mixup=use_mixup,
            mixup_alpha=0.4,
            use_specaugment=use_specaugment,
        )

        val_loss, metrics = validate(model=model, dataloader=val_loader, criterion=criterion, device=device)

        # Learning Rate Scheduler step
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']*100:.2f}% | Macro-F1: {metrics['macro_f1']:.4f}"
        )

        # Checkpoint Best Model
        is_best = metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = metrics["macro_f1"]

        save_checkpoint(
            state={
                "epoch": epoch,
                "exp_id": exp_id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            is_best=is_best,
            save_dir=os.path.join(output_dir, exp_id),
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })

    # Save training history log CSV
    log_df = pd.DataFrame(history)
    log_df.to_csv(os.path.join(output_dir, exp_id, "training_log.csv"), index=False)
    print(f"\nExperiment {exp_id} Complete. Best Validation Macro-F1: {best_val_f1:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bioacoustic Species Identification Model")
    parser.add_argument("--exp_id", type=str, default="E6", help="Experiment ID from experiments_matrix.json (e.g. E6)")
    args = parser.parse_args()

    run_experiment(exp_id=args.exp_id)
