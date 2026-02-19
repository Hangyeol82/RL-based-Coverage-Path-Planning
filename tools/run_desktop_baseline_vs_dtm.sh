#!/usr/bin/env bash
set -euo pipefail

# Run baseline vs DTM PPO training with identical settings.
#
# Usage example:
#   PYTHON=/path/to/python \
#   TOTAL_TIMESTEPS=50000 \
#   MAP_FILE=map/indoor_seed101.txt \
#   bash tools/run_desktop_baseline_vs_dtm.sh

PYTHON_BIN="${PYTHON:-python}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
SEED="${SEED:-101}"
MAP_FILE="${MAP_FILE:-map/indoor_seed101.txt}"
MAP_SIZE="${MAP_SIZE:-64}"
SENSOR_RANGE="${SENSOR_RANGE:-2}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-2000}"
N_STEPS="${N_STEPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_EPOCHS="${N_EPOCHS:-4}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="learning/checkpoints/rl/desktop_${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --map-source file
  --map-file "${MAP_FILE}"
  --map-size "${MAP_SIZE}"
  --sensor-range "${SENSOR_RANGE}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --seed "${SEED}"
  --n-steps "${N_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --n-epochs "${N_EPOCHS}"
)

echo "[INFO] output dir: ${OUT_DIR}"
echo "[INFO] baseline training..."
"${PYTHON_BIN}" run_ppo_sb3.py \
  "${COMMON_ARGS[@]}" \
  --save-model "${OUT_DIR}/ppo_baseline" \
  --save-breakdown-json "${LOG_DIR}/baseline.json" \
  --save-breakdown-csv "${LOG_DIR}/baseline.csv"

echo "[INFO] dtm training..."
"${PYTHON_BIN}" run_ppo_sb3.py \
  "${COMMON_ARGS[@]}" \
  --include-dtm \
  --save-model "${OUT_DIR}/ppo_dtm" \
  --save-breakdown-json "${LOG_DIR}/dtm.json" \
  --save-breakdown-csv "${LOG_DIR}/dtm.csv"

echo "[INFO] comparing logs..."
"${PYTHON_BIN}" tools/summarize_breakdown.py \
  --baseline "${LOG_DIR}/baseline.json" \
  --dtm "${LOG_DIR}/dtm.json"

echo "[DONE] baseline vs dtm run complete."
