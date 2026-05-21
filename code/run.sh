#!/usr/bin/env bash
set -ex

export PYTHONPATH=/code

echo "Setting up results directories..."
mkdir -p /results/models /results/logs /results/predictions

echo "Running the main training & inference pipeline..."

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
  num_workers=0

echo "Run completed successfully. Results are saved in /results."