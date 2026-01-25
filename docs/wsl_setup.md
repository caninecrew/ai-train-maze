# WSL / Ubuntu CLI Setup (VS Code)

This repo works well in WSL. Follow these steps from your Ubuntu shell.

## 1) Open in VS Code (WSL)
```bash
cd /path/to/ai-train-base
code .
```

## 2) Python + venv
```bash
python -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies
```bash
pip install -r requirements.txt
```

## 4) Run the maze workflow locally
Convert the maze:
```bash
python scripts/mazes/convert_png_to_grid.py --png assets/mazes/maze_001.png --out data/mazes/maze_001/maze_001
```

Train:
```bash
export MAZE_ID=maze_001
export MAZE_MAX_STEPS=3600
python train.py --game maze --config configs/maze.yaml
```

## 5) Common issues
- If `code .` fails, install the VS Code Remote WSL extension.
- If Tkinter is missing for `maze_game.py`, install it with: `sudo apt-get install python3-tk`.
