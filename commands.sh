#!/bin/bash
# =====================================================================
# StarCraft Vision Project: Consolidated Experiment & Execution Commands
# =====================================================================

# Replay IDs Reference:
# fold1: 275 1725 3613 4520 4664
# fold2: 1559 1628 2351 6219 11251
# fold3: 36 212 438 522 1660
# extra: 6254, 3529 3972 7191 7950 7970 9105 9301 9795

# =====================================================================
# 0. Quick Lightweight Test Commands (Low-Spec Client Workstation)
# =====================================================================
# --- 1) CenterNet ---
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=centernet batch_size=2 max_epoch=1 num_workers=1 dataset=fold1_sample"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=centernet batch_size=2 max_epoch=1 num_workers=1 dataset=fold1_sample seed=123"

# --- 2) Deformable DETR ---
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=deformable_detr batch_size=2 max_epoch=1 num_workers=1 dataset=fold1_sample"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=deformable_detr batch_size=2 max_epoch=1 num_workers=1 dataset=fold1_sample seed=123"

# --- 3) Mask R-CNN ---
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=maskrcnn batch_size=2 max_epoch=1 num_workers=1 dataset=fold1_sample"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=maskrcnn batch_size=2 max_epoch=1 num_workers=1 dataset=fold1_sample seed=123"

# --- 4) RT-DETR ---
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=rtdetr batch_size=2 max_epoch=1 num_workers=1 dataset=fold1_sample"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=rtdetr batch_size=2 max_epoch=1 num_workers=1 dataset=fold1_sample seed=123"

# --- Single-Replay Inference Test Example ---
NVIDIA_VISIBLE_DEVICES=0 make inference ARGS="--model-name centernet --model-number 1 --replays 36 --batch-size 2 --cuda"

# =====================================================================
# 1. Data Preprocessing
# =====================================================================
# Preprocess input frames
make preprocess_input ARGS="--replays 36 212 438 522 1660 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 1559 1628 2351 6219 11251 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 275 1725 3613 4520 4664 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 6254 3529 3972 7191 7950 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 7970 9105 9301 9795 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 6254 2351 275 1725 3613 4520 4664 3529 3972 7191 7950 7970 9105 9301 9795 --include-components worker ground air building vision neutral resource terrain"

# Preprocess target labels
make preprocess_label ARGS="--replays 36 212 438 522 1660 --method all_correct"
make preprocess_label ARGS="--replays 1559 1628 2351 6219 11251 --method all_correct"
make preprocess_label ARGS="--replays 275 1725 3613 4520 4664 --method all_correct"
make preprocess_label ARGS="--replays 6254 3529 3972 7191 7950 --method all_correct"
make preprocess_label ARGS="--replays 7970 9105 9301 9795 --method all_correct"
make preprocess_label ARGS="--replays 36 212 438 522 1660 1559 1628 2351 6219 11251 275 1725 3613 4520 4664 6254 3529 3972 7191 7950 7970 9105 9301 9795 --method all_correct"

# =====================================================================
# 2. Data Prechecking & KBRS Caching
# =====================================================================
make precheck ARGS=" \
  --root_dir /workspace/data/label/dst \
  --label_method all_correct \
  --replays 36,212,438,522,1660,1559,1628,2351,6219,11251,275,1725,3613,4520,4664 \
  --verbose \
"

make precheck ARGS=" \
  --root_dir /workspace/data/input/dst \
  --replays 36,212,438,522,1660,1559,1628,2351,6219,11251,275,1725,3613,4520,4664 \
  --mmap \
  --max_abs_warn 1000000 \
  --workers 32 \
  --chunksize 64 \
"

NVIDIA_VISIBLE_DEVICES=0 make cache ARGS=" \
  cache \
  --replays 275 1725 3613 4520 4664 1559 1628 2351 6219 11251 36 212 438 522 1660 \
  --num-workers 16 \
  --sample-ratio 1.0 \
  --log-level none \
"

# =====================================================================
# 3. Model Training & Pipeline Runs (Single Executions)
# =====================================================================
# Mask R-CNN Vanilla & KBRS
NVIDIA_VISIBLE_DEVICES=0 make run ARGS=" \
  architecture=maskrcnn \
  kbrs=disabled \
  dataset=fold1_sample \
  seed=123 \
"

NVIDIA_VISIBLE_DEVICES=0 make run ARGS=" \
  architecture=maskrcnn \
  kbrs=enabled \
  dataset=fold1_sample \
  seed=123 \
  kbrs_loss=kbrs025 \
"

# RT-DETR Vanilla & KBRS
NVIDIA_VISIBLE_DEVICES=0 make run ARGS=" \
  architecture=rtdetr \
  kbrs=disabled \
  dataset=fold1_sample \
  seed=123 \
  kbrs_loss=kbrs025 \
"

NVIDIA_VISIBLE_DEVICES=0 make run ARGS=" \
  architecture=rtdetr \
  kbrs=enabled \
  dataset=fold1_sample \
  seed=123 \
  kbrs_loss=kbrs025 \
"

# CenterNet Pure Base Model (Density Peak Disabled)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=centernet use_density_peak=false"

# CenterNet + Density Peak Head (Custom Extension Enabled)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=centernet use_density_peak=true"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS=" \
  architecture=centernet \
  use_density_peak=true \
  dataset=fold1 \
  seed=123 \
"

# Deformable DETR Pure Base Model (Probabilistic Query Disabled)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=deformable_detr use_probabilistic_query=false"

# Deformable DETR + Probabilistic Latent Query (Custom Extension Enabled)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=deformable_detr use_probabilistic_query=true"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS=" \
  architecture=deformable_detr \
  use_probabilistic_query=true \
  dataset=fold1 \
  seed=123 \
"

# Short Parameter Key Aliases Examples (arch, ds, win, lr, batch, loss, score)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="arch=centernet ds=fold1 win=4 lr=0.005 batch=16 seed=123"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="arch=deformable_detr ds=fold1 win=4 lr=0.001 batch=8 seed=123"

# Explicit Synology Chat Notification Enable Flag
ENABLE_SYNOLOGY_CHAT=true NVIDIA_VISIBLE_DEVICES=0 make train ARGS="arch=centernet ds=fold1"

# =====================================================================
# 4. Fold & Seed Sweeps
# =====================================================================
# Vanilla 3-fold seed sweep (3-fold x 3 seed = 9 runs)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="-m \
  dataset=fold1,fold2,fold3 \
  model=vanilla \
  seed=123,456,789 \
"

# KBRS 3-fold seed sweep (3-fold x 3 seed = 9 runs)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="-m \
  dataset=fold1,fold2,fold3 \
  model=kbrs \
  seed=123,456,789 \
  kbrs_loss=kbrs025 \
  kbrs_score=base_score \
"

# =====================================================================
# 5. KBRS Ablation Sweeps
# =====================================================================
# KBRS Loss Weight Ablation
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs050,kbrs075,kbrs100 \
  kbrs_score=base_score \
"

# KBRS Density Score Ablation
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=density/000,density/010,density/020,density/040,density/050,density/060,density/070,density/080,density/090 \
"

# KBRS Centeredness Score Ablation
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=centeredness/000,centeredness/010,centeredness/020,centeredness/040,centeredness/050,centeredness/060,centeredness/070,centeredness/080,centeredness/090 \
"

# KBRS Mixture Score Ablation
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=mixture/050,mixture/100,mixture/150,mixture/450,mixture/500 \
"

# Comprehensive Full Ablation Sweep (Loss weights & score variants)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs050,kbrs075,kbrs100 \
  kbrs_score=base_score \
"

NVIDIA_VISIBLE_DEVICES=0 make run ARGS="-m \
  dataset=fold2,fold3 \
  model=kbrs \
  seed=123,456,789 \
  kbrs_loss=kbrs025,kbrs050,kbrs075,kbrs100 \
  kbrs_score=base_score \
"

# =====================================================================
# 6. Label Estimation Runs
# =====================================================================
# Ground Truth Estimation across folds
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode gt --replays 275 1725 3613 4520 4664 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode gt --replays 1559 1628 2351 6219 11251 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode gt --replays 36 212 438 522 1660 --skip-kbrs"

# Vanilla Model Estimation (Folds 1, 2, 3)
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold1_s123_20251201_072032 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold1_s456_20251202_003952 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold1_s789_20251202_163217 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold2_s123_20251203_081925 --epoch 30 --replays 1559 1628 2351 6219 11251 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold2_s456_20251204_002252 --epoch 30 --replays 1559 1628 2351 6219 11251 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold2_s789_20251204_163313 --epoch 30 --replays 1559 1628 2351 6219 11251 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold3_s123_20251205_081306 --epoch 30 --replays 36 212 438 522 1660 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold3_s456_20251205_235535 --epoch 30 --replays 36 212 438 522 1660 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_vanilla_fold3_s789_20251206_161212 --epoch 30 --replays 36 212 438 522 1660 --skip-kbrs"

# KBRS Model Estimation (Folds 1, 2, 3)
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_base_score_20251219_080334 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold1_s456_kbrs025_base_score_20260101_175816 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold1_s789_kbrs025_base_score_20251230_004200 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold2_s123_kbrs025_base_score_20251222_080550 --epoch 30 --replays 1559 1628 2351 6219 11251 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold2_s456_kbrs025_base_score_20251231_031613 --epoch 30 --replays 1559 1628 2351 6219 11251 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold2_s789_kbrs025_base_score_20251231_215319 --epoch 30 --replays 1559 1628 2351 6219 11251 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold3_s123_kbrs025_base_score_20251230_083127 --epoch 30 --replays 36 212 438 522 1660 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold3_s456_kbrs025_base_score_20251231_151012 --epoch 30 --replays 36 212 438 522 1660 --skip-kbrs"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name all_correct_win4_kbrs_fold3_s789_kbrs025_base_score_20260102_163643 --epoch 30 --replays 36 212 438 522 1660 --skip-kbrs"

# =====================================================================
# 7. Replay Lookup Operations
# =====================================================================
# GT Label Lookup
NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 275 1725 3613 4520 4664 1559 1628 2351 6219 11251 36 212 438 522 1660 \
  --label-source gt \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 16 \
  --log-level log \
"

# Predicted Label Lookup
NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 275 1725 3613 4520 4664 \
  --label-source pred \
  --id-string all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_500_20260117_032417 \
  --epoch 30 \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 16 \
  --log-level log \
"

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name maskrcnn_win4_vanilla_fold1_s123_20251201_072032 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs --label-method all_correct"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name maskrcnn_win4_kbrs_fold1_s123_kbrs025_base_score_20260203_055642 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs --label-method all_correct"

NVIDIA_VISIBLE_DEVICES=0 make inference ARGS="--model-name deformable_detr_vanilla_win1_fold1_s123_20260818_071132 --model-number 30 --replays 275 1725 3613 4520 4664 --label-method all_correct --cuda"

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name deformable_detr_vanilla_win1_fold1_s123_20260818_071132 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs --label-method all_correct"
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="--mode model --model-name centernet_vanilla_win1_fold1_s123_20260818_071116 --epoch 30 --replays 275 1725 3613 4520 4664 --skip-kbrs --label-method all_correct"