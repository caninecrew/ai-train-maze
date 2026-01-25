# Pong AI Trainer

Train PPO agents on a custom Pong environment built with pygame and Stable Baselines3.

## What is in this repo
- `pong.py`: the playable Pong environment (human vs human, or headless auto-tracking).
- `train_pong_ppo.py`: the main training script for PPO agents.
- `eval_pong.py`: evaluate one or more saved models and compare results.
- `dashboard.py`: live metrics dashboard and comparisons.
- `configs/`: optional YAML/JSON configs and profiles for training.
- `tests/`: unit tests plus one slow smoke test that runs a tiny training loop.

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

## Run the game
- Demo the environment: `python pong.py` (controls: W/S left, Up/Down right; auto-tracks when headless).

## Train a model
- Basic training: `python train_pong_ppo.py --config configs/example.yaml`
- Override with CLI flags: `--train-timesteps 200000 --checkpoint-interval 1 --seed 0 --n-envs 4 --iterations-per-set 2`

Artifacts and outputs:
- `models/`: saved checkpoints and `_latest` model.
- `logs/`: run logs, metrics CSV, and JSON/HTML reports.
- `videos/`: combined and per-model training clips (if enabled).

## Evaluate a model
```
python eval_pong.py --model-path models/ppo_pong_custom_latest.zip --episodes 5
```

Optional flags:
- `--render` to visualize
- `--output-csv logs/eval.csv` to save a report
- `--compare models/a.zip models/b.zip --plot-path logs/compare.png`

## Common recipes
- Quick smoke (headless, short): `python train_pong_ppo.py --profile quick --max-cycles 1 --dry-run`
- 1-2 minute videos: `python train_pong_ppo.py --video-steps 3600 --max-video-seconds 120 --target-fps 30 --individual-videos`
- GPU profile: `python train_pong_ppo.py --profile gpu --iterations-per-set 2 --n-envs 16 --stream-tensorboard`
- Status check: `python train_pong_ppo.py --status`
- Live dashboard: `python dashboard.py` then open `http://127.0.0.1:8000`
- Arcade launcher: run `scripts/launch_arcade.bat`

## Tests
- Quick checks: `python -m pytest tests/test_pong_env.py`
- Training pipeline tests (skip the slow test): `python -m pytest tests/test_train_pipeline.py -m "not slow"`
- Include slow training smoke test: `python -m pytest tests/test_train_pipeline.py -m slow`

## CI artifacts
- GitHub Actions keeps all `logs/` in the repo for the dashboard.
- Only the newest `models/*.zip` and newest `videos/*.mp4` are kept; older ones are deleted each run.
- Outputs are committed back to the repo (no artifact upload).

## Troubleshooting
- Progress bars: install `pip install rich tqdm` (or `pip install stable-baselines3[extra]`) to enable progress output.
- Headless pygame: ensure `SDL_VIDEODRIVER=dummy` is respected (default when not rendering). On Linux servers install `libsdl2-dev` packages; on macOS use `brew install sdl2 sdl2_image`.
- Stable Baselines3 and torch: CPU-only installs work; for GPU, set `--device cuda` and ensure CUDA-enabled torch is installed.
- Resume training: keep `_latest` checkpoints; rerun training with the same `model_dir` to continue.
