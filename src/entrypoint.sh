#!/bin/bash
set -e  # fail fast

echo "[1/5] Generating input channels..."
python src/preprocessing/make_channels.py \
    --replays 212 213 214 \
    --src-dir /workspace/data/src \
    --dst-dir /workspace/data/dst/channels \
    --interval 2

echo "[2/5] Generating VPDS labels..."
python src/preprocessing/make_viewport_labels.py \
    --replays 212 213 214 \
    --method all_correct \
    --src-dir /workspace/data/src/vpds \
    --dst-dir /workspace/data/dst/labels

echo "[3/5] Creating paired (input, label) .npy..."
python src/preprocessing/make_paired_data.py \
    --replays 212 213 214 \
    --method all_correct \
    --output channel \
    --data-dir /workspace/data/dst \
    --result-dir /workspace/data/paired

echo "[4/5] Training model..."
python src/main.py \
    --load_dir /workspace/data/paired \
    --training 212 213 214 \
    --mode point2_labels \
    --num_classes 2 \
    --window_size 4 \
    --max_epoch 20 \
    --log_save_dir /workspace/saved_models/jht_pipeline

echo "[✔] All stages completed."
