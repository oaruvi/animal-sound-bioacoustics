# Bioacoustic Species Identification in Neotropical Soundscapes: A Deep Transfer Learning Framework

Official experimental repository and artifact suite for the machine learning species identification framework evaluated on 206 Neotropical taxa from the Central Andes of Colombia (El Silencio Biological Reserve / BirdCLEF+ 2025 dataset).

---

## 1. Overview & Research Objectives

Passive Acoustic Monitoring (PAM) provides an autonomous, non-invasive framework for tracking biodiversity across Neotropical sanctuaries. However, processing large-scale unannotated soundscapes requires automated computational tools capable of addressing severe taxonomic class imbalance, audio data leakage, and low-latency field execution.

This repository provides the complete, end-to-end implementation of a deep transfer learning framework for multi-taxonomic acoustic species identification. Key research highlights include:

* **Comprehensive Multi-Taxonomic Corpus**: Evaluation across 206 species spanning four taxonomic classes (Birds, Amphibians, Mammals, and Insects) using 28,564 field-collected audio recordings.
* **Recording-Level Leakage Prevention**: Implementation of a strict recording-level stratified partitioning protocol that avoids source-recording overlap between training and validation splits.
* **Systematic 18-Configuration Benchmark**: Experimental search space evaluating three deep neural backbones (EfficientNet-B1, ResNet50 V2, and PANNs CNN14) combined with Focal Loss ($\gamma = 2.0$), Class Weighting, Mixup augmentation ($\alpha = 0.4$), and Stochastic Weight Averaging (SWA).
* **Optimal Model Performance**: The top-performing configuration (`EfficientNet-B1 + Focal Loss + Mixup + SWA`) achieves **95.02% validation accuracy** and a **Macro-F1 score of 0.8412** on the recording-level holdout partition.
* **End-to-End Operational Architecture**: Deployment of *AnimalSound*, a client-server ecosystem integrating a high-concurrency FastAPI RESTful backend, a cross-platform Flutter mobile client, and TensorFlow Lite (TFLite) optimization for field inference.

---

## 2. Directory Layout

The repository is organized into modular directories following the CRISP-DM methodology:

```
bioacoustic-species-id/
├── README.md                      # Primary repository documentation
├── environment.yml                # Conda environment specification
├── configs/                       # Configuration parameters for audio and experiments
│   ├── preprocessing_config.json  # Audio signal transformation parameters
│   └── experiments_matrix.json    # Full specification for configurations E1–E6, R1–R6, P1–P6
├── data_preprocessing/            # Data ingestion, cleaning, and windowing
│   ├── 01_eda_and_cleaning.py     # Metadata deduplication and taxonomic merging
│   ├── 02_audio_preprocessing.py  # 6-stage Mel-spectrogram transformation pipeline
│   └── 03_recording_split.py      # Stratified recording-level train/validation partitioner
├── models/                        # Neural network backbones and loss functions
│   ├── backbones.py               # EfficientNet-B1, ResNet50, and PANNs CNN14 constructors
│   └── losses.py                  # Focal Loss and class-weighting implementations
├── training/                      # Training workflows and scheduling
│   ├── train_pipeline.py          # Master training loop with SWA, Mixup, and CosineLR
│   └── utils.py                   # Checkpoint management and epoch logging
├── evaluation/                    # Validation and disaggregated performance analysis
│   ├── evaluate_metrics.py        # Global accuracy, Macro-F1, and loss tracking
│   └── taxonomic_breakdown.py     # Disaggregated evaluation across Aves, Amphibia, Mammalia, Insecta
├── backend_api/                   # RESTful inference server
│   └── main.py                    # FastAPI server exposing POST /api/animal/analyze
└── mobile_client/                 # Mobile deployment and edge conversion
    ├── export_tflite.py           # TFLite model quantization script
    └── README_mobile.md           # Specifications for the Flutter AnimalSound application
```

---

## 3. Dataset Specification & Taxonomic Distribution

The evaluation dataset comprises 28,564 audio recordings in `.ogg` format, representing 206 distinct animal species collected in Neotropical habitats. Metadata processing involved cleaning 28,155 redundant metadata entries to yield a consolidated set of 409 core metadata records mapped directly to acoustic source files.

To evaluate algorithmic resilience against extreme class imbalance, performance is disaggregated across four target taxonomic groups:

| Taxonomic Group | Class Name | Species Count | Sample Count | Dataset Share (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Birds** | Aves | 192 | 27,648 | 96.79% |
| **Amphibians** | Amphibia | 6 | 583 | 2.04% |
| **Mammals** | Mammalia | 5 | 178 | 0.62% |
| **Insects** | Insecta | 3 | 155 | 0.54% |
| **Total** | **4 Groups** | **206** | **28,564** | **100.00%** |

---

## 4. Audio Preprocessing Pipeline

Field audio recordings ranging from 3 to 300 seconds are processed through a standardized six-stage pipeline to convert raw 1D acoustic pressure signals into $224 \times 224 \times 3$ RGB Mel-spectrogram tensors:

1. **Resampling**: Signals are resampled to $f_s = 32,000\text{ Hz}$ ($32\text{ kHz}$), supporting frequencies up to $16\text{ kHz}$ to capture Neotropical vocalization harmonics ($0.5\text{ kHz} - 14\text{ kHz}$).
2. **Fixed-Length Windowing**: Continuous signals are segmented into non-overlapping $5$-second windows ($160,000$ discrete samples per segment), with zero-padding applied to shorter files.
3. **Short-Time Fourier Transform (STFT)**: Computed using a Hann window of $N_{\text{fft}} = 2048$ samples and a hop length of $H = 512$ samples ($75\%$ overlap).
4. **Mel Filterbank Projection**: Spectral power is mapped onto $M = 128$ Mel frequency bands spanning $f_{\text{min}} = 50\text{ Hz}$ to $f_{\text{max}} = 14,000\text{ Hz}$.
5. **Logarithmic Compression & Normalization**: Dynamic range is compressed using $S_{\text{dB}} = 10 \log_{10}(S_{\text{Mel}} + 10^{-6})$ and min-max scaled to the interval $[0, 1]$.
6. **Spatial Resizing & Channel Replication**: Spectrograms are resized to $224 \times 224$ pixels and replicated across three identical RGB channels ($224 \times 224 \times 3$).

---

## 5. Experimental Benchmark Matrix

A systematic search space of 18 experimental configurations (6 per backbone architecture) was executed to identify optimal convergence conditions under extreme class imbalance.

### Complete 18-Configuration Specification

| Config ID | Neural Backbone | Loss Function & Class Weighting | Optimizer & Scheduler | Regularization & Augmentation | Validation Accuracy |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **E1** | **EfficientNet-B1** | **Focal Loss ($\gamma=2.0$)** | **AdamW + CosineLR** | **Mixup ($\alpha=0.4$) + SWA** | **95.02%** |
| E2 | EfficientNet-B1 | Weighted CE ($w_c \in [0.1, 10]$) | AdamW + CosineLR | CBAM Attention + Refine Block | 88.00% |
| E3 | EfficientNet-B1 | Effective Samples ($\beta=0.99$) | AdamW + CosineLR | Label Smoothing ($\epsilon=0.1$) | 93.95% |
| E4 | EfficientNet-B1 | Weighted CE ($w_c \in [0.1, 10]$) | AdamW + OneCycleLR | ImageNet Norm + Grad Clipping | 82.91% |
| E5 | EfficientNet-B1 | Standard Cross-Entropy | AdamW + CosineLR | Selective Layer Freezing | 87.53% |
| E6 | EfficientNet-B1 | Standard Cross-Entropy | SGD + Cosine WarmRestarts | Mixup ($\alpha=0.4$) + Label Smooth | 78.00% |
| R1 | ResNet50 V2 | Weighted CE ($w_c \in [0.1, 10]$) | AdamW + CosineLR | Dropout ($p=0.3$) + Grad Accumulation | 94.33% |
| R2 | ResNet50 V2 | Sqrt Inverse ($w_c \in [0.2, 5]$) | AdamW + OneCycleLR | Layer 1–2 Freeze + Label Smooth | 87.79% |
| R3 | ResNet50 V2 | Sqrt Inverse ($w_c \in [0.2, 5]$) | AdamW + OneCycleLR | Stratified Subsampling (50% Data) | 0.53% |
| R4 | ResNet50 V2 | Sqrt Inverse + Label Smooth | AdamW + OneCycleLR | Hybrid Medium Conv Head | 93.60% |
| R5 | ResNet50 V2 | Focal Loss ($\gamma=2.0$) | Discriminative LR + CosineLR | Mixup ($\alpha=0.4$) + SWA | 94.68% |
| R6 | ResNet50 V2 | Sqrt Inverse ($w_c \in [0.2, 5]$) | AdamW + CosineLR | Layer 1–2 Freeze (25% Data) | 79.79% |
| P1 | PANNs CNN14 | Standard Cross-Entropy | AdamW + OneCycleLR | Aggressive 6/8 Module Freeze | 61.02% |
| P2 | PANNs CNN14 | Class-Weighted CE | AdamW + StepLR | Progressive Layer Unfreezing | 57.07% |
| P3 | PANNs CNN14 | Focal Loss ($\gamma=2.0$) | Discriminative LR + CosineLR | Multi-tier Layer Grouping | 87.12% |
| P4 | PANNs CNN14 | Standard Cross-Entropy | AdamW + Cosine WarmRestarts | Mixup ($\alpha=0.4$) (25% Data) | 53.64% |
| P5 | PANNs CNN14 | Focal Loss ($\gamma=2.0$) | AdamW + Warmup Cosine Decay | Grad Accumulation (35% Data) | 71.75% |
| P6 | PANNs CNN14 | Class-Weighted CE | AdamW + SWA Scheduler | SpecAugment + Weight Averaging | 64.08% |

### Disaggregated Taxonomic Performance (Optimal Pipeline E1)

Applying Focal Loss ($\gamma = 2.0$), Mixup ($\alpha = 0.4$), and SWA substantially improves recall for severely underrepresented non-avian taxa compared to standard Cross-Entropy:

| Taxonomic Group | Species Count | CE Precision | CE Recall | Optimal Precision (E1) | Optimal Recall (E1) | Recall Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Birds (Aves)** | 192 | 0.9620 | 0.9580 | 0.9580 | 0.9610 | +0.3% |
| **Amphibians** | 6 | 0.7410 | 0.6200 | **0.8650** | **0.8420** | **+22.2%** |
| **Mammals** | 5 | 0.6820 | 0.5410 | **0.8120** | **0.7950** | **+25.4%** |
| **Insects** | 3 | 0.6150 | 0.4820 | **0.7640** | **0.7380** | **+25.6%** |

---

## 6. System Architecture & Deployment (*AnimalSound*)

The optimal EfficientNet-B1 engine is operationalized into *AnimalSound*, a client-server mobile software ecosystem:

* **Backend RESTful API (FastAPI)**: The server exposes the endpoint `POST /api/animal/analyze`, accepting field recordings in OGG, WAV, MP3, FLAC, and M4A formats (up to 10 MB). The API processes audio in memory through the identical 6-stage Mel-spectrogram pipeline, executes EfficientNet-B1 inference, and returns a JSON payload with top-$k$ species predictions, confidence scores, and taxonomic metadata.
* **Mobile Client (Flutter)**: The cross-platform mobile application manages device microphone access, records ambient soundscapes, and renders a dynamic time-frequency spectrogram visualization. Field researchers can select specific spectrogram regions to isolate vocalization phrases before triggering remote inference.
* **TFLite Edge Integration**: The client integrates TensorFlow Lite bindings for quantized on-device execution in off-grid field environments.

---

## 7. Setup & Execution Instructions

### Prerequisites & Environment Setup

Create and activate the Conda environment using the provided specification:

```bash
conda env create -f environment.yml
conda activate bioacoustic-id
```

### Data Preprocessing & Splitting

1. Process metadata and clean duplicate records:
   ```bash
   python data_preprocessing/01_eda_and_cleaning.py
   ```
2. Generate standardized Mel-spectrogram tensors:
   ```bash
   python data_preprocessing/02_audio_preprocessing.py --config configs/preprocessing_config.json
   ```
3. Generate stratified recording-level train/validation splits:
   ```bash
   python data_preprocessing/03_recording_split.py
   ```

### Model Training & Evaluation

To execute the top-performing configuration (E1):

```bash
python training/train_pipeline.py --config_id E1
```

To evaluate trained checkpoints and compute disaggregated taxonomic performance:

```bash
python evaluation/taxonomic_breakdown.py --checkpoint checkpoints/efficientnet_b1_optimal.pth
```

### Launching the Backend API

Start the FastAPI inference backend server:

```bash
cd backend_api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 8. Citation & Academic Reference

If you utilize this framework, dataset split protocols, or model architectures in your research, please cite the underlying study:

```bibtex
@mastersthesis{Lincango2026Thesis,
  author       = {Lincango-Simba{\~n}a, Betsy Bel{\'e}n},
  title        = {Identificaci{\'o}n ac{\'u}stica de especies a partir de audio, utilizando algoritmos de aprendizaje autom{\'a}tico},
  school       = {Universidad T{\'e}cnica Particular de Loja (UTPL)},
  year         = {2026},
  address      = {Loja, Ecuador},
  type         = {Master's Thesis}
}
```

```bibtex
@inproceedings{LincangoCSCI2026,
  author       = {Lincango-Simba{\~n}a, Betsy Bel{\'e}n and Ruiz-Vivanco, Omar Alexander},
  title        = {Bioacoustic Species Identification in Neotropical Soundscapes: A Deep Transfer Learning Framework},
  booktitle    = {Proceedings of the International Conference on Computational Science and Computational Intelligence (CSCI)},
  publisher    = {Springer Nature},
  year         = {2026}
}
```

