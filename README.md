# Mobile-CSLR: Continuous Sign Language Recognition with MobileViT + BiLSTM

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: PHOENIX-2014](https://img.shields.io/badge/Dataset-PHOENIX--2014-green)](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)

A lightweight, mobile-friendly Continuous Sign Language Recognition (CSLR) system trained on the **RWTH-PHOENIX-Weather 2014** benchmark. The model combines Apple's **MobileViT-Small** as a spatial feature extractor with a **Bidirectional LSTM** sequence decoder, trained end-to-end using dual **CTC loss**.

---

## 🏆 Results

| Split | WER (%) |
|-------|---------|
| Test (best checkpoint) | **~34.84** |

> Trained for 20 epochs on a single **Nvidia Tesla T4** GPU (Kaggle).

---

## 🏗️ Model Architecture

```
Input Video (B, T, C, H, W)
        │
        ▼
  MobileViT-Small          ← apple/mobilevit-small (HuggingFace)
  (frame-level encoder)
        │  pooler_output → (B*T, 640)
        ▼
  LayerNorm + reshape → (B, T, 640)
        │
        ▼
  Temporal Conv1D           ← kernel=5, stride=1, MaxPool1D(2)
  (Conv1d → BN → ReLU → Dropout → MaxPool1d)
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
  Bidirectional LSTM         Auxiliary FC → CTC logits
  (2 layers, hidden=1024)    (auxiliary CTC head)
        │
        ▼
  FC Output → CTC logits
        │
        ▼
  Combined CTC Loss (main + auxiliary)
```
![Model Architecture](Mobile%20ViT.jpg)

**Key design choices:**
- **MobileViT-Small** is used as a fully fine-tuned vision backbone.
- **Auxiliary CTC loss** on the post-conv features helps gradient flow to the backbone.
- **Temporal augmentation** (`TemporalRescale`) during training randomly sub/super-samples video frames.
- **Mixed precision** (AMP + GradScaler) for training efficiency.
- **Gradient clipping** (`max_norm=1.0`) for stability.

---

## 📦 Dataset

**RWTH-PHOENIX-Weather 2014** — the standard CSLR benchmark for German Sign Language (DGS).

| Split | Sequences |
|-------|-----------|
| Train | 5,672 |
| Dev   | 540   |
| Test  | 629   |

Download from the [official PHOENIX page](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).

Expected directory layout after download:

```
/workspace/datasets/phoenix/
├── annotations/manual/
│   ├── train.corpus.csv
│   ├── dev.corpus.csv
│   └── test.corpus.csv
└── fullFrame-210x260px/
    ├── train/
    ├── dev/
    └── test/
```

Update the paths at the top of the notebook to match your local setup.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download PHOENIX-2014 dataset

Follow instructions at: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

### 3. Update dataset paths

In the notebook, update these variables to point to your data:

```python
train_csv = "/path/to/phoenix/annotations/manual/train.corpus.csv"
test_csv  = "/path/to/phoenix/annotations/manual/test.corpus.csv"
dev_csv   = "/path/to/phoenix/annotations/manual/dev.corpus.csv"

train_paths = "/path/to/phoenix/fullFrame-210x260px/train"
test_paths  = "/path/to/phoenix/fullFrame-210x260px/test"
dev_paths   = "/path/to/phoenix/fullFrame-210x260px/dev"
```

### 4. Run training

Open and run [`mobile-vit-lstm-updated-loss.ipynb`](mobile-vit-lstm-updated-loss.ipynb).

The notebook is self-contained and runs on a single GPU. It was originally developed and executed on **Kaggle** with a T4 GPU.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | `apple/mobilevit-small` |
| Backbone output dim | 640 |
| LSTM hidden size | 1024 |
| LSTM layers | 2 (bidirectional) |
| Dropout | 0.1 |
| Max frames | 96 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-3 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs | 40 (best checkpoint saved) |
| Loss | CTC (blank=0) + auxiliary CTC |
| AMP | ✅ |
| Gradient clip | 1.0 |

---

## 📁 Repository Structure

```
Mobile-CSLR/
├── mobile-vit-lstm-updated-loss.ipynb   # Main training notebook
├── Mobile ViT.jpg                        # Architecture diagram
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Git ignore rules
├── LICENSE                               # MIT License
└── README.md                             # This file
```

---

## 📈 Training Curve

| Epoch | Train Loss | Train WER | Test Loss | Test WER |
|-------|-----------|-----------|-----------|----------|
| 1     | 9.17      | 91.30%    | 7.16      | 80.68%   |
| 5     | 3.87      | 43.71%    | 4.50      | 40.85%   |
| 10    | 2.43      | 28.14%    | 4.11      | 38.07%   |
| 15    | 1.60      | 17.40%    | 4.12      | 35.52%   |
| 20    | 1.25      | 12.04%    | 4.08      | 35.14%   |

---

## 💾 Checkpoints

Checkpoints are saved automatically during training as:

```
Mobile_ViT_LSTM_{epoch}_epochs_{test_wer:.2f}_wer.pth
```

Each checkpoint contains:
- `model_state_dict`
- `optimizer` state
- `scheduler` state
- `train_wer`, `test_wer`, `train_loss`, `test_loss`, `epoch`

To resume training, uncomment and update the checkpoint loading block in the notebook.

---

## 🔧 Data Augmentation

| Augmentation | Applied to |
|---|---|
| Random crop (256→224) | Train frames |
| Center crop (256→224) | Test/Dev frames |
| Random horizontal flip | Train videos |
| TemporalRescale (±20%) | Train videos |
| FPS sub-sampling (15–21 fps from 25 fps) | All splits |

---

## 📚 References

- [RWTH-PHOENIX-Weather 2014 Dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)
- [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (Apple)](https://arxiv.org/abs/2110.02178)
- [Connectionist Temporal Classification (CTC)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [SubUNets: End-to-end Hand Shape and Continuous Sign Language Recognition](https://arxiv.org/abs/1802.08073)

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> **Note:** The PHOENIX-2014 dataset has its own license terms. Please refer to the dataset page for redistribution and usage restrictions.