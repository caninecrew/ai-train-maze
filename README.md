# AI Training Base

A game-agnostic PPO training base with a modular game registry. Pong remains as the reference game adapter, but the repo is organized so you can add new games quickly without rewriting the training loop.

## What's in this repo
- `train.py`: generic PPO training loop with profiles, checkpoints, metrics, videos, and reports.
- `eval.py`: evaluate one or more checkpoints for any registered game.
- `dashboard.py`: live metrics dashboard that reads `logs/` outputs.
- `games/`: game adapters that expose gym-compatible environments.
- `pong.py`: the example game environment (pygame).
- `tests/`: smoke tests and environment checks.

Compatibility wrappers:
- `train_pong_ppo.py`: forwards to `train.py`.
- `eval_pong.py`: forwards to `eval.py`.

## Requirements
- Python 3.9+
- Recommended: use a virtual environment and install dependencies with `pip install -r requirements.txt`.

## Quickstart (venv)
```
python -m venv .venv
.\.venv\Scripts\activate   # or source .venv/bin/activate on *nix
pip install -r requirements.txt
```

If you prefer a one-liner setup after cloning, run the bootstrap scripts:
- Windows (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1`
- macOS/Linux: `bash scripts/setup_env.sh`

## List available games
```
python train.py --list-games
```

## Train a model
```
python train.py --game pong --config configs/pong.yaml
```

Override with CLI flags:
```
--train-timesteps 200000 --checkpoint-interval 1 --seed 0 --n-envs 4 --iterations-per-set 2
```

Artifacts and outputs:
- `models/`: saved checkpoints and `_latest` models.
- `logs/`: run logs, metrics CSV, and JSON/HTML reports.
- `videos/`: combined and per-model training clips (if enabled).

## Evaluate a model
```
python eval.py --game pong --model-path models/ppo_pong_custom_latest.zip --episodes 5
```

Optional flags:
- `--render` to visualize
- `--output-csv logs/eval.csv` to save a report
- `--compare models/a.zip models/b.zip --plot-path logs/compare.png`

## Dashboard
```
python dashboard.py
```
Then open `http://127.0.0.1:8000`.

## Common recipes
- Quick smoke (headless, short): `python train.py --game pong --profile quick --max-cycles 1 --dry-run`
- 1-2 minute videos: `python train.py --game pong --video-steps 3600 --max-video-seconds 120 --target-fps 30 --individual-videos`
- GPU profile: `python train.py --game pong --profile gpu --iterations-per-set 2 --n-envs 16 --stream-tensorboard`
- Status check: `python train.py --status`

## Add a new game
1) Create a gym-compatible environment (or wrapper) for your game.
2) Add a new adapter in `games/` that returns the env via `make_env`.
3) Register the adapter in `games/registry.py`.
4) Optionally add extra metrics in the adapter's `extra_metrics`.
5) Add a config in `configs/` for your new game.

## Tests
- Quick checks: `python -m pytest tests/test_pong_env.py`
- Training pipeline tests (skip the slow test): `python -m pytest tests/test_train_pipeline.py -m "not slow"`
- Include slow training smoke test: `python -m pytest tests/test_train_pipeline.py -m slow`

## Troubleshooting
- Progress bars: install `pip install rich tqdm` (or `pip install stable-baselines3[extra]`) to enable progress output.
- Headless pygame: ensure `SDL_VIDEODRIVER=dummy` is respected (default when not rendering). On Linux servers install `libsdl2-dev` packages; on macOS use `brew install sdl2 sdl2_image`.
- Stable Baselines3 and torch: CPU-only installs work; for GPU, set `--device cuda` and ensure CUDA-enabled torch is installed.
- Resume training: keep `_latest` checkpoints; rerun training with the same `model_dir` to continue.
