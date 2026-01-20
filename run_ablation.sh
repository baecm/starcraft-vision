#!/bin/bash
set -e

GPU=1

ABLATIONS_LIST=(
  density/000 density/010 density/020 density/040 density/050 density/060 density/070 density/080 density/090  
  centeredness/000 centeredness/010 centeredness/020 centeredness/040 centeredness/050 centeredness/060 centeredness/070 centeredness/080 centeredness/090 
  mixture/000 mixture/050 mixture/100 mixture/150 mixture/200 mixture/250 mixture/300 mixture/350 mixture/400 mixture/450 mixture/500 
  density_only/010 density_only/020 density_only/040 density_only/060 density_only/080 density_only/090 
  centeredness_only/010 centeredness_only/020 centeredness_only/040 centeredness_only/060 centeredness_only/080 centeredness_only/090 
  mixture_only/050 mixture_only/100 mixture_only/150 mixture_only/200 mixture_only/250 mixture_only/300 mixture_only/350 mixture_only/400 mixture_only/450 mixture_only/500 
)
ABLATIONS=$(IFS=,; echo "${ABLATIONS_LIST[*]}")

## LOSS WEIGHT ABLATION
# fold1, 123 loss weights025 is done
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs050,kbrs075,kbrs100 \
  kbrs_score=base_score \
  "

NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=456,789 \
  kbrs_loss=kbrs025,kbrs050,kbrs075,kbrs100 \
  kbrs_score=base_score \
  "

NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold2,fold3 \
  model=kbrs \
  seed=123,456,789 \
  kbrs_loss=kbrs025,kbrs050,kbrs075,kbrs100 \
  kbrs_score=base_score \
  "

# fold1, 123 is already done
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=456,789 \
  kbrs_loss=kbrs025 \
  kbrs_score=\"$ABLATIONS\" \
  "

# fold2 and fold3
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold2,fold3 \
  model=kbrs \
  seed=123,456,789 \
  kbrs_loss=kbrs025 \
  kbrs_score=\"$ABLATIONS\" \
  "

