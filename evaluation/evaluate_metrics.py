"""
Global Evaluation & Benchmarking Module
=======================================
Evaluates trained bioacoustic classification models on the strict recording-level holdout partition.
Computes overall accuracy, Macro-F1, Weighted-F1, Log-Loss, and per-class confusion metrics.

Usage:
    python evaluate_metrics.py --config configs/preprocessing_config.json \
                               --model_path models/checkpoints/best_model.pth \
                               --val_csv data/val_split.csv \
                               --output_dir evaluation/results
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, log_loss

# Import local modules
from backbones import get_model
from losses import FocalLoss
from audio_preprocessing import BioacousticDataset, get_audio_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bioacoustic Model Metrics")
    parser.add_argument("--config", type=str, default="configs/preprocessing_config.json", help="Path to preprocessing config")
    parser.add_argument("--model_name", type=str, default="efficientnet_b1", help="Architecture name")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pth", help="Path to model weights")
    parser.add_argument("--val_csv", type=str, default="data/val_split.csv", help="Path to validation CSV")
    parser.add_argument("--audio_dir", type=str, default="data/raw/audio", help="Directory containing audio files")
    parser.add_argument("--num_classes", type=int, default=206, help="Number of species classes")
    parser.add_argument("--batch_size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--output_dir", type=str, default="evaluation/results", help="Directory to save evaluation reports")
    return parser.parse_args()


def run_evaluation():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Evaluation on Device: {device}")

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Prepare Validation DataLoader
    val_transform = get_audio_transforms(config, is_train=False)
    val_dataset = BioacousticDataset(
        csv_path=args.val_csv,
        audio_dir=args.audio_dir,
        config=config,
        transform=val_transform,
        is_train=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Instantiate and Load Model Weights
    model = get_model(architecture=args.model_name, num_classes=args.num_classes, pretrained=False)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"[+] Loaded weights from {args.model_path}")
    else:
        print(f"[!] Warning: Model checkpoint '{args.model_path}' not found. Evaluating uninitialized model.")

    model = model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0

    print("[*] Performing inference on validation set...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)
    all_preds = np.argmax(all_probs, axis=1)

    # Compute Global Metrics
    total_val_loss = running_loss / len(val_dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
    overall_logloss = log_loss(all_targets, all_probs, labels=list(range(args.num_classes)))

    print("\n" + "=" * 60)
    print(f" GLOBAL EVALUATION RESULTS (Recording-Level Holdout)")
    print("=" * 60)
    print(f" Validation Accuracy:    {acc * 100:.2f}%")
    print(f" Macro-F1 Score:         {macro_f1:.4f}")
    print(f" Weighted-F1 Score:      {weighted_f1:.4f}")
    print(f" Validation Loss:        {total_val_loss:.4f}")
    print(f" Multiclass Log-Loss:    {overall_logloss:.4f}")
    print("=" * 60)

    # Export Structured Results
    metrics_summary = {
        "architecture": args.model_name,
        "validation_samples": len(val_dataset),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "validation_loss": float(total_val_loss),
        "log_loss": float(overall_logloss)
    }

    summary_file = os.path.join(args.output_dir, f"{args.model_name}_global_metrics.json")
    with open(summary_file, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[+] Saved global metrics summary to: {summary_file}")

    # Export Per-Class Detailed Classification Report
    cls_report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    cls_report_df = pd.DataFrame(cls_report).transpose()
    report_file = os.path.join(args.output_dir, f"{args.model_name}_per_class_report.csv")
    cls_report_df.to_csv(report_file)
    print(f"[+] Saved per-class report to: {report_file}")


if __name__ == "__main__":
    run_evaluation()
