#!/bin/bash
set -e

# ./run_estimate.sh
# ./run_estimate_kbrs.sh
# ./run_estimate_loss_weight.sh
# ./run_estimate_density.sh
./run_estimate_density_only.sh
./run_estimate_centeredness.sh
./run_estimate_centeredness_only.sh
./run_estimate_mixture.sh
./run_estimate_mixture_only.sh