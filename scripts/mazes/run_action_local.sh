#!/usr/bin/env bash
set -euo pipefail

# Local runner that mirrors .github/workflows/maze_train.yml with faster defaults.

MAZE_ID="${1:-maze_001}"
ROWS="${2:-60}"
COLS="${3:-60}"
TRAIN_GAME="${4:-maze}"
MAX_CYCLES="${5:-6}"
TRAIN_ARGS="${6:-}"
EARLY_STOP_PATIENCE="${7:-9999}"
ITERATIONS_PER_SET="${8:-2}"
MAX_STEPS="${9:-2400}"
TRAIN_MODE="${10:-true}"
TRAIN_AUTO="${11:-true}"
TRAIN_GOAL="${12:-}"
TRAIN_GOAL_FRACTION="${13:-0.35}"
TRAIN_TIMESTEPS="${14:-40000}"
N_ENVS="${15:-8}"
EVAL_EPISODES="${16:-3}"
DEVICE="${17:-cpu}"
NO_RESUME="${18:-false}"

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MAZE_MAX_STEPS="$MAX_STEPS"
export MAZE_TRAIN_MODE="$TRAIN_MODE"
export MAZE_TRAIN_AUTO="$TRAIN_AUTO"
export MAZE_TRAIN_GOAL="$TRAIN_GOAL"
export MAZE_TRAIN_GOAL_FRACTION="$TRAIN_GOAL_FRACTION"
export MAZE_MOVE_BONUS="${MAZE_MOVE_BONUS:-0.0}"
export MAZE_COOKIE_BONUS="${MAZE_COOKIE_BONUS:-0.03}"
export MAZE_SHAPING_COEF="${MAZE_SHAPING_COEF:-0.2}"
export MAZE_NOVELTY_BONUS="${MAZE_NOVELTY_BONUS:-0.02}"
export MAZE_BACKTRACK_PENALTY="${MAZE_BACKTRACK_PENALTY:--0.4}"
export TRAIN_DISABLE_GIT="true"
export TRAIN_WORKER_TIMEOUT="1200"
export TRAIN_NO_MULTIPROC="true"

mkdir -p videos models logs

png=""
if [ -n "$MAZE_ID" ]; then
  png="assets/mazes/$MAZE_ID.png"
else
  png="$(ls -t assets/mazes/maze_*.png 2>/dev/null | head -n 1 || true)"
fi
if [ -z "$png" ]; then
  echo "No maze PNGs found in assets/mazes."
  exit 1
fi

maze_id="$(basename "$png" .png)"
cfg="assets/mazes/$maze_id.json"
out_dir="data/mazes/$maze_id"
mkdir -p "$out_dir"

cfg_arg=()
if [ -f "$cfg" ]; then
  cfg_arg=(--config "$cfg")
fi

python3 scripts/mazes/convert_png_to_grid.py \
  --png "$png" \
  --rows "$ROWS" \
  --cols "$COLS" \
  --out "$out_dir/$maze_id" \
  "${cfg_arg[@]}"

resume_arg=()
if [ "${NO_RESUME,,}" != "true" ]; then
  best_resume=""
  if [ -f logs/metrics.csv ]; then
    best_resume="$(python3 scripts/mazes/find_best_checkpoint.py 2>/dev/null || true)"
  fi
  if [ -n "$best_resume" ]; then
    resume_arg=(--resume-from "$best_resume")
    echo "Using best checkpoint: $best_resume"
  fi
fi

if [ -z "$TRAIN_ARGS" ]; then
  TRAIN_ARGS="--n-steps 512 --batch-size 1024 --n-epochs 3 --video-steps 600 --max-video-seconds 20 --target-fps 30"
fi

python3 train.py \
  --game "$TRAIN_GAME" \
  --max-cycles "$MAX_CYCLES" \
  --iterations-per-set "$ITERATIONS_PER_SET" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --train-timesteps "$TRAIN_TIMESTEPS" \
  --n-envs "$N_ENVS" \
  --eval-episodes "$EVAL_EPISODES" \
  --device "$DEVICE" \
  "${resume_arg[@]}" \
  $TRAIN_ARGS
