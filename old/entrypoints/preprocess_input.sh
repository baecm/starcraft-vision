#!/bin/bash
set -e  # fail fast

echo "Generating input channels..."

python src/preprocessing/input.py "$@"

echo "Preprocessing completed."
