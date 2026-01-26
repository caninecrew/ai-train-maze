# AI Training Base

A game-agnostic PPO training base with a modular adapter registry. This repo is a starting point for training an AI on your own environment.

## Maze-solving pipeline
This repo can power a maze-solving AI system that separates three concerns: maze design, fast training/solving, and visual playback.

What it does:
- Maze input (human-friendly): design a maze as a PNG so you can edit walls visually.
- Conversion (training-friendly): convert the PNG into a compact grid (wall/open) and cache it.
- Fast solving/training: run on the fixed-size grid to keep the state space small; use shaping, imitation, or BFS/A* for near-instant paths.
- Visual output: render a video by overlaying the agent path on the original PNG and export an MP4.

Deliverables:
- `assets/mazes/maze_001.png`: the visual maze you designed.
- `data/mazes/maze_001/maze_001_grid.npy` or `.txt`: the AI-readable grid.
- `data/mazes/maze_001/maze_001_meta.json`: grid size plus start/goal.
- `outputs/mazes/maze_001/maze_001_run.mp4`: the character moving through the original maze image.

Best practices and naming convention:
- See `docs/maze_best_practices.md` for the full pipeline, thresholds, and scaling guidance.
- Maze IDs use `maze_###` (e.g., `maze_001`) and stay consistent across assets/data/outputs.
- Optional per-maze config can live at `assets/mazes/maze_###.json` (rows/cols, thresholds, wall_ratio).
- GitHub Actions: run `maze-convert-and-train` manually to convert the latest maze and train; adjust `max_cycles` and `train_args` as needed.
- Local training: use `scripts/mazes/train_local.ps1` (Windows) or `scripts/mazes/train_local.sh` (macOS/Linux). They auto-resume from the best checkpoint in `logs/metrics.csv` when present.

## What's in this repo
- `train.py`: generic PPO training loop with profiles, checkpoints, metrics, videos, and reports.
- `eval.py`: evaluate one or more checkpoints for any registered game.
- `dashboard.py`: live metrics dashboard that reads `logs/` outputs.
- `games/`: game adapters that expose gym-compatible environments.
- `docs/`: how-tos and adapter examples.
- `tests/`: smoke tests and environment checks.

## Requirements
- Python 3.9+
- Recommended: use a virtual environment and install dependencies with `pip install -r requirements.txt`.
- Minimal (conversion/render only): `pip install -r requirements-min.txt`.

## Quickstart (venv)
```
python -m venv .venv
.\.venv\Scripts\activate   # or source .venv/bin/activate on *nix
pip install -r requirements.txt
```

## WSL / Ubuntu CLI setup (VS Code)
If you already have Ubuntu CLI (WSL) installed:
1) Open the repo from WSL: `code .` inside the repo directory.
2) Create and activate a venv:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
3) Install deps: `pip install -r requirements.txt`
4) For a full walkthrough, see `docs/wsl_setup.md`.

If you prefer a one-liner setup after cloning, run the bootstrap scripts:
- Windows (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1`
- macOS/Linux: `bash scripts/setup_env.sh`

## List available games
```
python train.py --list-games
```

## Train a model
```
python train.py --game template --config configs/template.yaml
```

Maze training (uses latest converted maze by default):
```
python train.py --game maze --config configs/maze.yaml
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
python eval.py --game template --model-path models/ppo_template_latest.zip --episodes 5
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
- Quick smoke (headless, short): `python train.py --game template --profile quick --max-cycles 1 --dry-run`
- 1-2 minute videos: `python train.py --game template --video-steps 3600 --max-video-seconds 120 --target-fps 30 --individual-videos`
- GPU profile: `python train.py --game template --profile gpu --iterations-per-set 2 --n-envs 16 --stream-tensorboard`
- Status check: `python train.py --status`
- Export resolved config: `python train.py --export-config logs/resolved_config.json`
- Evo + PPO (maze): `python train.py --game maze --evo-first`
- Evolutionary maze swarm (ray sensors + A* scoring): `bash scripts/mazes/run_evo_local.sh`

## Add a new game
1) Create a gym-compatible environment (or wrapper) for your game.
2) Add a new adapter in `games/` that returns the env via `make_env`.
3) Register the adapter in `games/registry.py`.
4) Optionally add extra metrics in the adapter's `extra_metrics`.
5) Add a config in `configs/` for your new game.
6) See `docs/quick_guide.md` and `games/README.md` for examples.

## Tests
- Training pipeline tests (skip the slow test): `python -m pytest tests/test_train_pipeline.py -m "not slow"`
- Include slow training smoke test: `python -m pytest tests/test_train_pipeline.py -m slow`
- Adapter contract test: `python -m pytest tests/test_adapter_contract.py`

## Troubleshooting
- Progress bars: install `pip install rich tqdm` (or `pip install stable-baselines3[extra]`) to enable progress output.
- Headless pygame: ensure `SDL_VIDEODRIVER=dummy` is respected (default when not rendering). On Linux servers install `libsdl2-dev` packages; on macOS use `brew install sdl2 sdl2_image`.
- Stable Baselines3 and torch: CPU-only installs work; for GPU, set `--device cuda` and ensure CUDA-enabled torch is installed.
- Resume training: keep `_latest` checkpoints; rerun training with the same `model_dir` to continue.
