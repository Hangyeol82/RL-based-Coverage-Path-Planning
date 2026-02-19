#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_bc_ablation_seed5.sh --map-source random --map-stage 3
#   bash run_bc_ablation_seed5.sh --map-source indoor --indoor-rooms-rows 5 --indoor-rooms-cols 5
#   bash run_bc_ablation_seed5.sh --map-source file --map-file my_map.txt

SEEDS=(101 202 303 404 505)

for s in "${SEEDS[@]}"; do
  echo "===== SEED ${s} ====="
  python run_bc_quick_ablation.py \
    --seed "${s}" \
    --map-size 64 \
    --sensor-range 2 \
    --epochs 40 \
    --batch-size 64 \
    --max-steps 2500 \
    "$@" || exit 1
  echo
done
