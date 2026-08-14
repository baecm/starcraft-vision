# StarCraft Vision: Object Detection & Knowledge-Based Region Scoring

A Deep Learning Object Detection and Spatiotemporal Vision Pipeline designed for analyzing dense StarCraft replay battle sequences.

---

## 📌 Project Overview

This repository provides an end-to-end multi-frame object detection framework tailored for StarCraft vision analysis. It supports multiple neural architectures, temporal windowing, knowledge-based region scoring (KBRS), and a modular Docker-based workflow.

### Supported Model Architectures
- **Mask R-CNN**: Two-stage instance segmentation baseline.
- **RT-DETR (v1/v2)**: Real-time transformer-based detector.
- **CenterNet with Density Peak Head**: Anchor-free detector with density fields ($\rho$) and peak cutoff distances ($\delta$) to separate dense, overlapping unit clusters.
- **Deformable Video DETR with Probabilistic Latent Query**: Spatiotemporal deformable attention across frame windows combined with probabilistic query sampling ($z \sim \mathcal{N}(\mu, \sigma^2)$) for unit uncertainty modeling.

---

## 📁 Repository Structure

```
starcraft-vision/
├── conf/                     # Hydra configuration files (architecture, dataset, kbrs, loss, score)
├── infra/                    # Docker infrastructure & docker-compose services
├── src/                      # Model builders, training loop, dataset, pipeline, & evaluation
├── commands.sh               # Exhaustive script containing all replay experiment commands
├── Makefile                  # Build & execution automation commands
└── README.md                 # Project documentation
```

---

## 🐳 Docker Services (`infra/docker-compose.yml`)

The project uses modular `docker-compose` services with minimal required volume bindings:

| Service | Primary Role | Bound Volumes |
| :--- | :--- | :--- |
| `preprocessor` | Data & label preprocessing, dataset auditing | `src` (ro), `data` (rw) |
| `trainer` | Model training & sequential pipeline | `src` (ro), `conf` (ro), `data` (ro), `models` (rw), `results` (rw), `predictions` (rw) |
| `inferencer` | Model prediction & COCO export | `src` (ro), `data` (ro), `models` (ro), `predictions` (rw), `.torch_cache` |
| `evaluator` | IC metrics, evaluation & estimation | `src` (ro), `data` (rw), `models` (ro), `predictions` (ro), `results` (rw) |
| `debugger` | Interactive bash debugging | Full volume access |

---

## 🚀 Execution Guide (Representative Examples)

> [!TIP]
> For the complete exhaustive list of all specific replay IDs, fold sweeps, and ablation lists, refer to [commands.sh](file:///home/bcm/workspace/starcraft-vision/commands.sh).

### 1. Data Preprocessing & Auditing
```bash
# Preprocess input frame tensors & target labels
make preprocess_input ARGS="--replays 36 212 438 --include-components worker ground air building vision"
make preprocess_label ARGS="--replays 36 212 438 --method all_correct"

# Audit dataset integrity & precompute KBRS cache
make precheck ARGS="--root_dir /workspace/data/label/dst --label_method all_correct --replays 36,212,438 --verbose"
NVIDIA_VISIBLE_DEVICES=0 make cache ARGS="cache --replays 36 212 438 --num-workers 16"
```

### 2. Model Training & Pipeline Execution

```bash
# CenterNet (Density Peak Head)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=centernet"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=centernet dataset=fold1 seed=123"

# Deformable Video DETR (Probabilistic Latent Query)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=deformable_video_detr"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=deformable_video_detr dataset=fold1 seed=123"

# Mask R-CNN & RT-DETR
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=maskrcnn dataset=fold1_sample seed=123"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=rtdetr kbrs=enabled dataset=fold1_sample seed=123"
```

### 3. Sweeps & Ablations
```bash
# Multi-fold seed sweep
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="-m dataset=fold1,fold2,fold3 model=kbrs seed=123,456,789"

# KBRS Score Ablation (Density / Centeredness / Mixture)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m dataset=fold1 model=kbrs seed=123 kbrs_loss=kbrs025 kbrs_score=density/000,density/010,density/020"
```

### 4. Estimation & Replay Lookup
```bash
# Label Estimation (Ground Truth & Model Predictions)
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode gt --replays 275 1725 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name <MODEL_NAME> --epoch 30 --replays 275 1725"

# Replay Lookup
NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS="--replays 275 1725 --label-source pred --id-string <MODEL_NAME> --epoch 30"
```

---

## 🔔 Optional Environment Flags

### Synology Chat Notifications
Synology Chat notifications are disabled by default. Enable explicitly via `ENABLE_SYNOLOGY_CHAT=true`:

```bash
make train ARGS="architecture=centernet" ENABLE_SYNOLOGY_CHAT=true
```
