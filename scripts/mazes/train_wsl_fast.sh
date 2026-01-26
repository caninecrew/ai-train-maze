#!/usr/bin/env bash
set -euo pipefail

GAME="${1:-maze}"
MAX_CYCLES="${2:-6}"
TRAIN_ARGS="${3:-}"
MAZE_ID="${4:-}"
MAZE_MAX_STEPS="${5:-3600}"
MAZE_CELL_SIZE="${6:-8}"

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
if [ -n "$MAZE_ID" ]; then
  export MAZE_ID="$MAZE_ID"
fi
if [ -n "$MAZE_MAX_STEPS" ]; then
  export MAZE_MAX_STEPS="$MAZE_MAX_STEPS"
fi
if [ -n "$MAZE_CELL_SIZE" ]; then
  export MAZE_CELL_SIZE="$MAZE_CELL_SIZE"
fi

mkdir -p videos models logs

best=""
if [ -f logs/metrics.csv ]; then
  best="$(python scripts/mazes/find_best_checkpoint.py 2>/dev/null || true)"
fi

base_args=(--game "$GAME" --config configs/maze_wsl_cpu.yaml --max-cycles "$MAX_CYCLES")
if [ -n "$best" ]; then
  python train.py "${base_args[@]}" --resume-from "$best" $TRAIN_ARGS
else
  python train.py "${base_args[@]}" $TRAIN_ARGS
fi
