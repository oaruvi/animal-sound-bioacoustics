"""
03_recording_split.py
---------------------
Strict Recording-Level Stratified Partitioning Protocol.

This module enforces a strict recording-level split to avoid source-recording
overlap between training and validation sets, mitigating metric inflation caused
by continuous audio clip repetition and background noise memorization.

Author: Betsy Belén Lincango Simbaña (UTPL)
Project: Bioacoustic Species Identification in Neotropical Soundscapes
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def perform_recording_level_split(
    metadata_path: str = "data/metadata_cleaned.csv",
    output_dir: str = "data/processed",
    config_path: str = "configs/preprocessing_config.json",
    random_seed: int = 42
):
    """
    Splits the dataset ensuring zero file-level overlap between train and validation.
    
    Args:
        metadata_path (str): Path to cleaned metadata CSV.
        output_dir (str): Directory where train and val split CSVs will be saved.
        config_path (str): Path to preprocessing configuration JSON.
        random_seed (int): Seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration if available
    train_val_ratio = 0.8
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            train_val_ratio = config.get("dataset", {}).get("train_val_ratio", 0.8)

    logging.info(f"Loading cleaned metadata from {metadata_path}...")
    if not os.path.exists(metadata_path):
        # Generate synthetic/dummy structure if running standalone
        logging.warning("Metadata file not found. Creating a structural template execution.")
        np.random.seed(random_seed)
        n_samples = 28564
        df = pd.DataFrame({
            "recording_id": [f"REC_{i//10:05d}" for i in range(n_samples)],
            "clip_id": [f"CLIP_{i:05d}" for i in range(n_samples)],
            "primary_label": np.random.choice([f"sp_{j:03d}" for j in range(206)], size=n_samples),
            "taxonomic_group": np.random.choice(["Avian", "Amphibian", "Mammal", "Insect"], size=n_samples, p=[0.9679, 0.0204, 0.0062, 0.0054])
        })
    else:
        df = pd.read_csv(metadata_path)

    logging.info(f"Total clips to partition: {len(df)}")
    logging.info(f"Unique source recordings: {df['recording_id'].nunique()}")

    # Group by recording_id to enforce zero-overlap constraint
    sgkf = StratifiedGroupKFold(n_splits=int(1 / (1 - train_val_ratio)), shuffle=True, random_state=random_seed)
    
    groups = df["recording_id"]
    targets = df["primary_label"]

    train_idx, val_idx = next(sgkf.split(df, targets, groups))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)

    # Verification of Zero File-Level Overlap
    train_recordings = set(train_df["recording_id"])
    val_recordings = set(val_df["recording_id"])
    overlap = train_recordings.intersection(val_recordings)

    logging.info("=" * 60)
    logging.info("RECORDING-LEVEL PARTITIONING SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Training Clips:   {len(train_df)} ({len(train_df)/len(df)*100:.2f}%) | Unique Recordings: {len(train_recordings)}")
    logging.info(f"Validation Clips: {len(val_df)} ({len(val_df)/len(df)*100:.2f}%) | Unique Recordings: {len(val_recordings)}")
    logging.info(f"Source Overlap Count: {len(overlap)} (Zero Overlap Confirmed: {len(overlap) == 0})")
    logging.info("=" * 60)

    # Save split manifests
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logging.info(f"Saved training manifest -> {train_path}")
    logging.info(f"Saved validation manifest -> {val_path}")

    return train_df, val_df

if __name__ == "__main__":
    perform_recording_level_split()
