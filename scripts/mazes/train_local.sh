#!/usr/bin/env bash
set -euo pipefail

GAME="${1:-template}"
MAX_CYCLES="${2:-5}"
TRAIN_ARGS="${3:-}"

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

mkdir -p videos models logs

python train.py --game "$GAME" --max-cycles "$MAX_CYCLES" $TRAIN_ARGS
