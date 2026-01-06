#!/bin/bash
set -e

GPU=0

NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=mixture_only/200,mixture_only/250,mixture_only/350,mixture_only/400 \
  "

# NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
#   dataset=fold1 \
#   model=kbrs \
#   seed=123 \
#   kbrs_loss=kbrs025 \
#   kbrs_score=mixture_only/150,mixture_only/450,mixture_only/100,mixture_only/500,mixture_only/050, \
#   "