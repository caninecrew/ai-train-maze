#!/usr/bin/env bash
set -euo pipefail

GAME="${1:-template}"
MAX_CYCLES="${2:-5}"
TRAIN_ARGS="${3:-}"
MAZE_ID="${4:-}"

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
if [ -n "$MAZE_ID" ]; then
  export MAZE_ID="$MAZE_ID"
fi

mkdir -p videos models logs

python train.py --game "$GAME" --max-cycles "$MAX_CYCLES" $TRAIN_ARGS
