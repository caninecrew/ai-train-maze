# Maze AI Best Practices

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

## Maze input format and conversion
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

## Start/goal handling
- Store start and goal in metadata, not the image.
- `start = (row, col)` and `goal = (row, col)` live in `maze_meta.json`.

## Fast training/solving
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

## Video generation
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

## File layout and naming convention
Use a stable maze identifier with a 3-digit index:
- Format: `maze_###` (e.g., `maze_001`, `maze_002`)
- Keep the identifier consistent across assets, cached data, and outputs.

Recommended layout:
- `assets/mazes/maze_001.png`
- `assets/mazes/maze_001.json` (optional config overrides)
- `data/mazes/maze_001/maze_001_grid.npy`
- `data/mazes/maze_001/maze_001_grid.txt`
- `data/mazes/maze_001/maze_001_meta.json`
- `outputs/mazes/maze_001/maze_001_run.mp4`

Converter location:
- `scripts/mazes/convert_png_to_grid.py`

GitHub Actions automation:
- The `maze-convert-and-train` workflow runs conversion and training.
- You start it manually via workflow dispatch to keep outputs intentional.
- It converts the most recent maze by default, or uses the `maze_id` input when provided.
- Use `max_cycles` to control training length and `train_args` for extra flags.
- Optional per-maze settings live in `assets/mazes/maze_###.json` (e.g., rows/cols, threshold, wall_ratio).
- Set `MAZE_ID` in your environment to train a specific maze locally.

## Scalability strategy
- Keep grid size fixed to bound state space.
- Cache conversion and distance maps.
- Use fast planners or imitation when time is critical.
- If corridors degrade at fixed resolution: raise grid size to 80x80, then consider graph reduction if needed.

Summary:
PNG is for design and visualization; grid is for training and control. This keeps the workflow simple and fast while scaling to more complex mazes.
