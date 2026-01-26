#!/usr/bin/env bash
set -euo pipefail

MAZE_ID="${1:-maze_001}"
CYCLES="${2:-20}"
POPULATION="${3:-50}"
MAX_STEPS="${4:-2400}"
SENSOR_RANGE="${5:-20}"
TOP_K="${6:-10}"

mkdir -p logs models

python3 scripts/mazes/evo_train.py \
  --maze-id "$MAZE_ID" \
  --cycles "$CYCLES" \
  --population "$POPULATION" \
  --max-steps "$MAX_STEPS" \
  --sensor-range "$SENSOR_RANGE" \
  --top-k "$TOP_K" \
  --move-start-on-success
