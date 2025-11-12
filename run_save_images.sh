#!/bin/bash
# run_save_images.sh

# If you had: set -euo pipefail
# keep -e and -o pipefail, but avoid -u OR guard expansions.
set -eo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Headless plotting + import path (guard PYTHONPATH if unset)
export MPLBACKEND=Agg
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== Environment ==="
which python
python --version
pip --version
echo "==================="

python -u scratch/scripts/save_some_images.py

echo "✅ Done. Check: drive_samples.png, ph2_samples.png, weakph2_click_samples.png"
