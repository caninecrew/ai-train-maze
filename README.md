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
- `maze.png`: the visual maze you designed.
- `maze_grid.npy` or `.txt`: the AI-readable grid.
- `meta.json`: grid size plus start/goal.
- `run.mp4`: the character moving through the original maze image.

## Codex best practices for maze AI
Project objective:
Build a system where a maze is created visually as a PNG, converted into a training-friendly representation, solved or trained as quickly as possible, and then rendered back into a video showing a moving character solving the maze on top of the original PNG.

The system must support:
- Easy maze creation/modification by a human (graphical tools).
- Fast solving/training (low runtime).
- Clear visual outputs (video with moving character).
- Scalability to larger/more complex maze shapes later.

Best-practice architecture: separate logic from presentation.
Why:
- Training on pixels is slow, fragile, and unnecessary for maze navigation.
- A one-time conversion step produces a compact grid for fast training.
- The grid is authoritative for training/evaluation; the PNG is authoritative for design/visualization.

Maze input format and conversion best practices:
1) Use PNG as the maze design format:
   - Visual editing in external tools.
   - Easy to swap new mazes without code changes.
   - Clean final demo (video over original maze art).
   - Keep PNGs high-contrast (white corridors, dark walls); avoid heavy antialiasing.
2) Convert PNG -> grid using patch voting:
   - Divide the image into a target grid (e.g., 60x60).
   - For each grid cell, compute the fraction of white pixels.
   - Mark a cell open if white fraction exceeds a threshold.
   - Output encoding: 0 = open, 1 = wall.
3) Keep grid resolution fixed to bound training time:
   - Recommended default: 60x60.
   - 40x40 is faster, 80x80 preserves more detail.
   - Keep it consistent so the state space is bounded.
4) Cache conversion results:
   - `maze_grid.npy` for fast load.
   - `maze_grid.txt` for ASCII preview.
   - `maze_meta.json` for rows/cols + start/goal.

Start/goal handling best practices:
- Store start and goal in metadata, not the image.
- `start = (row, col)` and `goal = (row, col)` live in `maze_meta.json`.

Fast training/solving best practices:
1) Solve on the grid, not the PNG:
   - Discrete (row, col) states, 4-way actions, wall collisions rejected.
2) Prefer planners or supervised policies for speed:
   - A) BFS/A* for near-instant solutions.
   - B) Imitation learning with expert paths from BFS/A*.
   - C) RL with distance-to-goal shaping.
3) Reduce wasted exploration:
   - Step limits per episode.
   - Penalty for wall hits.
   - Small step cost.
   - Optional penalty for revisiting cells.

Video generation best practices:
1) Render on top of the original PNG.
2) Map grid to pixels:
   - x = (col + 0.5) * (image_width / cols)
   - y = (row + 0.5) * (image_height / rows)
3) Overlay a character efficiently:
   - Simple dot (most robust) or sprite PNG with transparency.
   - Copy background and draw the agent each frame.
   - Encode with MP4 (OpenCV or imageio).
4) Optional clarity improvements:
   - Trail line, success color change, pause on success, GIF export.

File/repo organization best practices:
- `assets/`: source art (maze PNG, optional sprite).
- `data/`: cached grid/meta outputs.
- `src/`: conversion, solver, renderer scripts.
- `outputs/`: videos and runs.

Scalability strategy:
- Keep grid size fixed to bound state space.
- Cache conversion and distance maps.
- Use fast planners or imitation when time is critical.
- If corridors degrade at fixed resolution: raise grid size to 80x80, then consider graph reduction if needed.

Summary:
PNG is for design and visualization; grid is for training and control. This keeps the workflow simple and fast while scaling to more complex mazes.

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
python train.py --game template --config configs/template.yaml
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
