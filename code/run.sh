#!/usr/bin/env bash
set -ex

export PYTHONPATH=/code

echo "Setting up results directories..."
mkdir -p /results/models /results/logs /results/predictions

# ---------------------------------------------------------
# Step 1. Train & Inference
# ---------------------------------------------------------
echo "1. Running the main training & inference pipeline..."
bash ./entrypoint.sh run \
  -m \
  dataset=fold1_sample \
  model=kbrs \
  mode=train_and_inference \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=base \
  kbrs_score.density=0.3 \
  kbrs_score.mixture=3.0 \
  kbrs_score.centeredness=0.3 \
  max_epoch=1 \
  num_workers=0 \
  +score_threshold=0.0

# ---------------------------------------------------------
# Step 2. find model and prediction directories
# ---------------------------------------------------------
echo "2. Finding the generated model directory..."
# retrieve the most recently created directory name in /results/predictions
MODEL_NAME=$(ls -t /results/predictions | head -n 1)
echo "Found model name: $MODEL_NAME"

# ---------------------------------------------------------
# Step 3. Evaluation (estimate.py)
# ---------------------------------------------------------
echo "3. Running evaluation and generating metrics..."
bash ./entrypoint.sh estimate \
  --mode model \
  --model-name "$MODEL_NAME" \
  --epoch 1 \
  --replays 1725 \
  --input-root /data/input/dst \
  --label-root /data/label/dst \
  --pred-root /results/predictions \
  --csv-out /results/final_metrics.csv

echo "Pipeline fully completed! Check /results/final_metrics.csv"