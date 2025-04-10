#!/bin/bash
set -e  # fail fast

echo "Generating input channels..."

python src/preprocessing/pair.py "$@"

echo "Preprocessing completed."
