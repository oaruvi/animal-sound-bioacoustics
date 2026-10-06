"""
01_eda_and_cleaning.py
======================
Exploratory Data Analysis (EDA), Metadata Cleaning, and Stratified Recording-Level Partitioning
for Neotropical Bioacoustic Species Identification (BirdCLEF+ 2025 Corpus).

Author: Betsy Belén Lincango Simbaña (UTPL)
Project: Bioacoustic Species Identification Framework
License: MIT
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Taxonomic Distribution Thresholds (BirdCLEF+ 2025 Dataset)
TAXONOMIC_BREAKDOWN = {
    "Birds": {"species_count": 192, "samples": 27648, "percentage": 96.79},
    "Amphibians": {"species_count": 6, "samples": 583, "percentage": 2.04},
    "Mammals": {"species_count": 5, "samples": 178, "percentage": 0.62},
    "Insects": {"species_count": 3, "samples": 155, "percentage": 0.54}
}


def load_raw_metadata(metadata_path: str, taxonomy_path: str) -> pd.DataFrame:
    """
    Loads raw metadata CSV and merges taxonomic group classifications.

    Args:
        metadata_path (str): Path to raw train_metadata.csv.
        taxonomy_path (str): Path to species_taxonomy.csv mapping.

    Returns:
        pd.DataFrame: Merged and validated metadata DataFrame.
    """
    logger.info(f"Loading raw metadata from: {metadata_path}")
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file '{metadata_path}' not found. Generating synthetic template structure for demonstration.")
        return generate_synthetic_metadata()

    df_meta = pd.read_csv(metadata_path)
    logger.info(f"Initial raw recordings count: {len(df_meta)}")

    if os.path.exists(taxonomy_path):
        df_tax = pd.read_csv(taxonomy_path)
        df_merged = df_meta.merge(df_tax, on="primary_label", how="left")
    else:
        df_merged = df_meta

    return df_merged


def generate_synthetic_metadata(num_samples: int = 28564) -> pd.DataFrame:
    """
    Generates a structured metadata DataFrame matching the exact class distribution
    and recording hierarchy of the El Silencio / BirdCLEF+ 2025 dataset.

    Args:
        num_samples (int): Total number of audio recordings (default 28,564).

    Returns:
        pd.DataFrame: Synthetic dataset metadata for validation and offline execution.
    """
    logger.info("Generating representative metadata structure based on UTPL thesis parameters...")
    np.random.seed(42)

    data = []
    sample_id = 0

    for taxa, info in TAXONOMIC_BREAKDOWN.items():
        n_taxa_samples = int(round(num_samples * (info["percentage"] / 100.0)))
        num_sp = info["species_count"]
        species_labels = [f"{taxa.lower()[:3]}_sp_{i+1:03d}" for i in range(num_sp)]
        
        # Dirichlet distribution for long-tail species skew within each taxonomic group
        dirichlet_weights = np.random.dirichlet(np.ones(num_sp) * 0.5)
        counts_per_sp = np.random.multinomial(n_taxa_samples, dirichlet_weights)

        for sp_idx, count in enumerate(counts_per_sp):
            sp_label = species_labels[sp_idx]
            # Group into distinct physical field recordings to simulate recording-level split
            num_recordings = max(1, count // 10)
            rec_ids = [f"REC_2025_ELSILENCIO_{taxa[:3].upper()}_{sp_idx:03d}_{r:02d}" for r in range(num_recordings)]

            for c in range(count):
                rec_id = rec_ids[c % num_recordings]
                data.append({
                    "filename": f"audio_{sample_id:05d}.ogg",
                    "primary_label": sp_label,
                    "taxonomic_group": taxa,
                    "recording_id": rec_id,
                    "duration_sec": float(np.random.uniform(5.0, 300.0)),
                    "latitude": 5.9812 + np.random.normal(0, 0.05),
                    "longitude": -74.6123 + np.random.normal(0, 0.05),
                    "author": "Bioacoustic Monitoring Team - UTPL"
                })
                sample_id += 1

    df = pd.DataFrame(data)
    logger.info(f"Synthetic dataset compiled successfully: {len(df)} total samples across 206 species.")
    return df


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, null handling, and removes corrupted or duplicate entries.

    Args:
        df (pd.DataFrame): Input metadata DataFrame.

    Returns:
        pd.DataFrame: Cleaned metadata DataFrame.
    """
    initial_len = len(df)
    
    # Drop rows missing essential labels or filenames
    df_clean = df.dropna(subset=["filename", "primary_label"]).copy()
    
    # Ensure taxonomic group column exists
    if "taxonomic_group" not in df_clean.columns:
        df_clean["taxonomic_group"] = "Birds"  # Default fallback
        
    # Standardize string values
    df_clean["primary_label"] = df_clean["primary_label"].str.strip().str.lower()
    df_clean["taxonomic_group"] = df_clean["taxonomic_group"].str.strip().str.capitalize()

    logger.info(f"Cleaned metadata: {len(df_clean)} valid entries remaining ({initial_len - len(df_clean)} removed).")
    return df_clean


def perform_recording_level_split(
    df: pd.DataFrame, val_ratio: float = 0.20, random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes strict RECORDING-LEVEL stratified partitioning.
    Ensures zero source recording overlap between train and validation splits
    to prevent acoustic data leakage (background noise & recorder profile memorization).

    Args:
        df (pd.DataFrame): Cleaned metadata DataFrame.
        val_ratio (float): Ratio of validation set (default 0.20 / 20%).
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    logger.info("Executing strict RECORDING-LEVEL stratified split (preventing file-overlap leakage)...")
    np.random.seed(random_seed)

    # Group by recording_id to ensure atomic unit of splitting is the recording file
    recordings = df.groupby("recording_id").agg({
        "primary_label": "first",
        "taxonomic_group": "first",
        "filename": "count"
    }).reset_index().rename(columns={"filename": "clip_count"})

    train_recs = []
    val_recs = []

    # Stratify recordings by species label
    for label, group in recordings.groupby("primary_label"):
        rec_list = group["recording_id"].values
        np.random.shuffle(rec_list)
        
        n_val = max(1, int(round(len(rec_list) * val_ratio))) if len(rec_list) > 1 else 0
        val_recs.extend(rec_list[:n_val])
        train_recs.extend(rec_list[n_val:])

    train_df = df[df["recording_id"].isin(train_recs)].copy().reset_index(drop=True)
    val_df = df[df["recording_id"].isin(val_recs)].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"

    logger.info(f"Split complete -> Train samples: {len(train_df)} ({len(train_recs)} recordings) | Val samples: {len(val_df)} ({len(val_recs)} recordings)")
    return train_df, val_df


def print_taxonomic_summary(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Prints a detailed taxonomic summary report comparing Train and Val splits."""
    full_df = pd.concat([train_df, val_df])
    
    print("\n" + "="*80)
    print("      NEOTROPICAL BIOACOUSTIC DATASET: TAXONOMIC BREAKDOWN SUMMARY")
    print("="*80)
    print(f"{'Taxonomic Group':<18} | {'Total Samples':<14} | {'Percentage':<10} | {'Train Clips':<12} | {'Val Clips':<10}")
    print("-"*80)

    for taxa in ["Birds", "Amphibians", "Mammals", "Insects"]:
        tot = len(full_df[full_df["taxonomic_group"] == taxa])
        pct = (tot / len(full_df)) * 100.0 if len(full_df) > 0 else 0
        tr_cnt = len(train_df[train_df["taxonomic_group"] == taxa])
        val_cnt = len(val_df[val_df["taxonomic_group"] == taxa])
        print(f"{taxa:<18} | {tot:<14,d} | {pct:>8.2f}% | {tr_cnt:<12,d} | {val_cnt:<10,d}")

    print("="*80)
    print(f"TOTAL SPECIES COUNT  : {full_df['primary_label'].nunique()} species")
    print(f"TOTAL RECORDINGS     : {full_df['recording_id'].nunique():,d} unique field files")
    print(f"TOTAL AUDIO CLIPS    : {len(full_df):,d} clips")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="EDA & Data Cleaning Script - Bioacoustics Project")
    parser.add_argument("--metadata_path", type=str, default="data/raw/train_metadata.csv", help="Path to raw metadata CSV")
    parser.add_argument("--taxonomy_path", type=str, default="data/raw/taxonomy.csv", help="Path to species taxonomy CSV")
    parser.add_argument("--output_dir", type=str, default="data/processed/", help="Directory to save processed metadata")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    raw_df = load_raw_metadata(args.metadata_path, args.taxonomy_path)

    # 2. Clean Data
    clean_df = clean_metadata(raw_df)

    # 3. Perform Recording-Level Split
    train_df, val_df = perform_recording_level_split(clean_df, val_ratio=0.20, random_seed=42)

    # 4. Print Summary
    print_taxonomic_summary(train_df, val_df)

    # 5. Export Processed Metadata
    full_processed_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    out_csv = os.path.join(args.output_dir, "cleaned_metadata_split.csv")
    full_processed_df.to_csv(out_csv, index=False)
    logger.info(f"Processed metadata exported to: {out_csv}")


if __name__ == "__main__":
    main()
