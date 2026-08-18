# StarCraft Vision: Object Detection & Knowledge-Based Region Scoring

A Deep Learning Object Detection and Spatiotemporal Vision Pipeline designed for analyzing dense StarCraft replay battle sequences.

---

## 📌 Project Overview

This repository provides an end-to-end multi-frame object detection framework tailored for StarCraft vision analysis. It supports multiple neural architectures, temporal windowing, knowledge-based region scoring (KBRS), and a modular Docker-based workflow.

### Supported Model Architectures & Reusable Plugins
- **Mask R-CNN**: Two-stage instance segmentation baseline (supports KBRS plugin).
- **RT-DETR (v1/v2)**: Real-time transformer-based detector (supports KBRS & Probabilistic Query plugins).
- **CenterNet (with Density Peak Head Plugin)**: Keypoint-based anchor-free detector with density fields ($\rho$) and peak cutoff distances ($\delta$) to separate dense, overlapping unit clusters.
- **Deformable Video DETR (with Probabilistic Latent Query Plugin)**: Spatiotemporal deformable attention across frame windows combined with probabilistic query sampling ($z \sim \mathcal{N}(\mu, \sigma^2)$) for unit uncertainty modeling.

---

## 📁 Repository Structure

```text
starcraft-vision/
├── conf/                         # Hydra configuration files (architecture, dataset, kbrs, loss)
│   └── architecture/             # Model & plugin configs (centernet, deformable_video_detr, maskrcnn, rtdetr)
├── infra/                        # Docker infrastructure, entrypoint.sh & docker-compose services
├── Makefile                      # Build & execution automation (dynamic UID/GID export)
├── commands.sh                   # Exhaustive script containing all replay experiment commands
│
└── src/                          # Modular Python Source Directory
    ├── train.py                  # Core Model Training Loop
    ├── inference.py              # Core Inference & Prediction Export (log-optimized tqdm)
    ├── evaluate.py               # Core Model Evaluation
    ├── estimate.py               # IC Estimation & Metric Calculation
    ├── pipeline.py               # Sequential Train & Inference Pipeline
    ├── custom_evaluator.py       # Metric Calculation Engine
    ├── cli.py                    # Shared Argument Parser
    ├── config.py                 # Global Configurations
    │
    ├── model/                    # Modular Model Framework
    │   ├── factory.py            # Unified Model Builder Factory (build_model)
    │   ├── backbones/            # Pure Model Architectures (centernet, deformable_detr, maskrcnn, rtdetr)
    │   ├── plugins/              # Reusable Component Plugins
    │   │   ├── kbrs/             # Universal KBRS Feature Map Scorer & Hook
    │   │   ├── density_peak.py   # Density Peak & Delta Head Plugin
    │   │   └── probabilistic_query.py # Gaussian Query & KL-Loss Plugin
    │   └── utils/                # Box Ops, Transforms & Feature Utilities
    │
    ├── dataset/                  # PyTorch Datasets & DataLoaders
    ├── preprocessing/            # Frame & Label Processing Pipeline
    │
    └── tools/                    # Auxiliary Diagnostic & Offline Tools
        ├── kbrs/                 # Offline KBRS Cache, Lookup & Profiling Tools
        │   ├── kbrs_cache.py
        │   ├── kbrs_lookup.py
        │   ├── kbrs_from_input.py
        │   └── profile_kbrs_scorer.py
        └── precheck/             # Replay & Label Audit Tools
            ├── precheck.py
            └── precheck_label.py
```

---

## 🔌 Modular Plugin System Summary

| Plugin | Primary Function | Target Backbones | Key Parameters / Config |
| :--- | :--- | :--- | :--- |
| **KBRS** (`plugins/kbrs/`) | Universal Feature Map Region Scorer & Guidance Loss | All Architectures (Mask-RCNN, RT-DETR, CenterNet, Deformable DETR) | `use_kbrs=true`, `kbrs_params`, `loss_kbrs=0.25` |
| **Density Peak Head** (`plugins/density_peak.py`) | Heatmap density & peak offset scaling for dense unit clusters | CenterNet | `use_density_peak=true`, `loss_density=0.5`, `loss_delta=0.5` |
| **Probabilistic Query** (`plugins/probabilistic_query.py`) | Gaussian query sampling ($z \sim \mathcal{N}(\mu, \sigma^2)$) & KL-Divergence loss | Deformable Video DETR, RT-DETR | `use_probabilistic_query=true`, `loss_kl=0.1` |

When building a model, `model.factory` logs a structured summary box displaying attached plugins:

```text
======================================================================
[MODEL BUILD] Architecture : CENTERNET
[MODEL BUILD] Input Chans  : 36 (Window Size: 4)
----------------------------------------------------------------------
[PLUGINS ATTACHED SUMMARY]
  - KBRS Plugin          : [DISABLED]
  - Density Peak Plugin  : [ENABLED]
  - Probabilistic Query  : [N/A]
======================================================================
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
NVIDIA_VISIBLE_DEVICES=0 make cache ARGS="--replays 36 212 438 --num-workers 16"
```

### 2. Model Training & Pipeline Execution

```bash
# CenterNet (with Density Peak Head)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=centernet architecture.use_density_peak=true"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=centernet dataset=fold1 seed=123"

# Deformable Video DETR (with Probabilistic Latent Query)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=deformable_video_detr architecture.use_probabilistic_query=true"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=deformable_video_detr dataset=fold1 seed=123"

# Mask R-CNN & RT-DETR with KBRS
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
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name <MODEL_NAME> --epoch 30 --replays 275 1725 --label-method all_correct"

# Replay Lookup
NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS="--replays 275 1725 --label-source pred --id-string <MODEL_NAME> --epoch 30"
```
