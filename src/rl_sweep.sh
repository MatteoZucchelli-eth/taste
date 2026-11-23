#!/usr/bin/env bash
# Step 2 beta sweep (CONTEXT §6g decision 5): one run per beta, very strong -> moderate, personal reward.
# Usage: bash src/rl_sweep.sh [hours per run] [beta ...]
set -u
H=${1:-2}; shift || true
BETAS=${@:-0.01 0.003 0.001 0.0003}
mkdir -p outputs/rl/logs
for b in $BETAS; do
  docker compose run --rm -T -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True taste \
    python src/rl_train.py --run personal_b$b --z user --beta $b --hours $H > outputs/rl/logs/personal_b$b.log 2>&1
done
