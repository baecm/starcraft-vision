#!/bin/bash
set -e

# 1) vanilla seed runs
./run_vanilla_seed_runs.sh

# 2) kbrs seed runs
./run_kbrs_seed_runs.sh

# 3) mixture ablation
./run_kbrs_mixture_ablation.sh

# 4) loss weight ablation
./run_kbrs_loss_weight_ablation.sh

# 5) density ablation
./run_kbrs_density_ablation.sh

# 6) centeredness ablation
./run_kbrs_centeredness_ablation.sh