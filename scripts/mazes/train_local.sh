#!/usr/bin/env bash
set -euo pipefail

GAME="${1:-template}"
MAX_CYCLES="${2:-5}"
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
if [ -n "$best" ]; then
  python train.py --game "$GAME" --max-cycles "$MAX_CYCLES" --resume-from "$best" $TRAIN_ARGS
else
  python train.py --game "$GAME" --max-cycles "$MAX_CYCLES" $TRAIN_ARGS
fi
