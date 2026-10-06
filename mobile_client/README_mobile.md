# AnimalSound — Client-Server & Mobile Ecosystem

`AnimalSound` is an end-to-end client-server software ecosystem engineered to bridge the gap between deep bioacoustic model training and real-time field deployment in Neotropical nature reserves.

---

## 1. System Architecture Overview

```text
[ Smartphone Mic / Audio File ]
               │
               ▼
┌──────────────────────────────┐
│    Flutter Mobile Client     │
│  - Spectrogram Selector      │
│  - Real-Time Audio Capture   │
└──────────────┬───────────────┘
               │  HTTP POST /api/animal/analyze
               ▼
┌──────────────────────────────┐
│     FastAPI REST Backend     │
│  - In-Memory 6-Stage Preproc │
│  - EfficientNet-B1 Model     │
└──────────────┬───────────────┘
               │  JSON Top-k Response
               ▼
┌──────────────────────────────┐
│  Species Identification Result│
│  (Confidence & Metadata)     │
└──────────────────────────────┘
```

---

## 2. Key Modules & Features

### A. Mobile Client Interface (Flutter)
* **Cross-Platform Support:** Engineered with Flutter for unified deployment on Android and iOS devices.
* **Audio Capture:** Requests smartphone microphone permissions to record ambient field soundscapes in `.wav` or `.ogg` format.
* **Interactive Spectrogram Range Selector:** Renders a dynamic time-frequency spectrogram visualization. Field researchers can drag selection handles over specific 5-second intervals to isolate target bird or amphibian vocalizations from ambient stream/wind noise before sending the request.
* **Top-$k$ Result Display:** Renders species rankings with common names, scientific names, confidence percentages, and taxonomic group icons (Aves, Amphibia, Mammalia, Insecta).

### B. RESTful API Backend (FastAPI)
* **Endpoint:** `POST /api/animal/analyze`
* **In-Memory Preprocessing:** Receives raw audio payloads up to 10 MB, executes 32 kHz resampling, 5-second windowing, 2048-sample STFT, 128-band Mel projection, log-dB compression, min-max scaling, and $224 \times 224 \times 3$ RGB tensor formatting in memory without disk write latency.
* **Inference Engine:** Evaluates the `EfficientNet-B1 + Focal Loss + Mixup + SWA` model and returns structured JSON predictions.

### C. Offline Inference Roadmap (TensorFlow Lite)
* **On-Device Inference:** For remote Neotropical reserves lacking cellular/Wi-Fi coverage, `export_tflite.py` converts trained PyTorch checkpoints into FP16-quantized TFLite models (~14 MB).
* **TFLite Flutter Integration:** Direct binding via the `tflite_flutter` package allows executing local inference directly on the mobile CPU/NPU.

---

## 3. Quick Start — API Backend

```bash
# Start FastAPI backend server
python backend_api/main.py

# Send test audio for inference
curl -X 'POST' \
  'http://localhost:8000/api/animal/analyze?top_k=5' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_bird.ogg;type=audio/ogg'
```
