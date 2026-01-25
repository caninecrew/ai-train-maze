"""
maze_game.py

Playable Maze Game (Tkinter) that loads your maze files:
- maze_###_grid.npy (preferred) OR maze_###_grid_xo.txt
- maze_###_meta.json (required)

Controls:
- Arrow keys or WASD to move
- R to reset
- N / P to load next/previous maze_id
- Esc to quit

Run:
  python maze_game.py --maze_dir ./mazes --maze_id 1 --cell_size 24 --prefer_npy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import math
import time
from typing import Tuple, Union, Optional, Dict, Any, List, Set, TYPE_CHECKING, cast

import numpy as np
try:
    import tkinter as tk
    from tkinter import messagebox
    tk = cast(Any, tk)
    messagebox = cast(Any, messagebox)
except Exception:  # pragma: no cover - optional GUI dependency
    tk = cast(Any, None)
    messagebox = cast(Any, None)

if TYPE_CHECKING:
    import tkinter as tk  # pragma: no cover


# ----------------------------
# Loading helpers
# ----------------------------

def _read_meta(meta_path: str) -> Dict[str, Any]:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for key in ("rows", "cols", "start", "goal"):
        if key not in meta:
            raise ValueError(f"Meta missing '{key}' in {meta_path}")

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    start = meta.get("start")
    goal = meta.get("goal")

    if start is not None:
        start = tuple(int(x) for x in start)
        if len(start) != 2:
            raise ValueError("Meta 'start' must be [row, col] or null.")
    if goal is not None:
        goal = tuple(int(x) for x in goal)
        if len(goal) != 2:
            raise ValueError("Meta 'goal' must be [row, col] or null.")

    meta["rows"] = rows
    meta["cols"] = cols
    meta["start"] = start
    meta["goal"] = goal
    return meta


def _read_grid_npy(path: str) -> np.ndarray:
    grid = np.load(path)
    if grid.ndim != 2:
        raise ValueError(f"Grid .npy must be 2D. Got shape={grid.shape} from {path}")

    grid = grid.astype(np.int8, copy=False)
    grid = np.where(grid != 0, 1, 0).astype(np.int8, copy=False)
    return grid


def _read_grid_xo_txt(path: str) -> np.ndarray:
    """
    Accepts:
      X or # as wall
      O or . as open
    Ignores whitespace.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    rows: List[List[int]] = []
    for ln in lines:
        compact = "".join(ch for ch in ln if not ch.isspace())
        if not compact:
            continue

        row: List[int] = []
        for ch in compact:
            if ch in ("X", "x", "#"):
                row.append(1)
            elif ch in ("O", "o", "."):
                row.append(0)
            else:
                raise ValueError(
                    f"Unrecognized grid char '{ch}' in {path}. Use X/O or #/."
                )
        rows.append(row)

    if not rows:
        raise ValueError(f"Grid txt appears empty: {path}")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"Grid txt rows must be same length: {path}")

    return np.array(rows, dtype=np.int8)


def _resolve_paths(maze_dir: str, maze_id: Union[int, str]) -> Tuple[str, str, str]:
    maze_id_str = str(maze_id).zfill(3)
    meta_path = os.path.join(maze_dir, f"maze_{maze_id_str}_meta.json")
    npy_path = os.path.join(maze_dir, f"maze_{maze_id_str}_grid.npy")
    txt_path = os.path.join(maze_dir, f"maze_{maze_id_str}_grid_xo.txt")
    if os.path.exists(meta_path):
        return meta_path, npy_path, txt_path

    subdir = os.path.join(maze_dir, f"maze_{maze_id_str}")
    meta_path = os.path.join(subdir, f"maze_{maze_id_str}_meta.json")
    npy_path = os.path.join(subdir, f"maze_{maze_id_str}_grid.npy")
    txt_path = os.path.join(subdir, f"maze_{maze_id_str}_grid_xo.txt")
    return meta_path, npy_path, txt_path


def load_maze_files(
    maze_dir: str,
    maze_id: Union[int, str],
    prefer_npy: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    maze_id_str = str(maze_id).zfill(3)
    meta_path, npy_path, txt_path = _resolve_paths(maze_dir, maze_id_str)

    meta = _read_meta(meta_path)

    grid: Optional[np.ndarray] = None
    if prefer_npy and os.path.exists(npy_path):
        grid = _read_grid_npy(npy_path)
    elif os.path.exists(txt_path):
        grid = _read_grid_xo_txt(txt_path)
    elif os.path.exists(npy_path):
        grid = _read_grid_npy(npy_path)
    else:
        raise FileNotFoundError(
            f"No grid file found for maze_{maze_id_str}. Expected {npy_path} or {txt_path}"
        )

    rows, cols = meta["rows"], meta["cols"]
    if grid.shape != (rows, cols):
        raise ValueError(
            f"Grid shape {grid.shape} does not match meta rows/cols ({rows},{cols})."
        )

    start = meta["start"]
    goal = meta["goal"]

    if start is None or goal is None:
        start, goal = _auto_pick_start_goal(grid, start, goal)
        meta["start"] = start
        meta["goal"] = goal

    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
        raise ValueError("Start/goal must be within grid bounds.")

    if grid[sr, sc] == 1:
        raise ValueError("Start position is on a wall cell.")
    if grid[gr, gc] == 1:
        raise ValueError("Goal position is on a wall cell.")

    return grid, meta


def render_grid_frame(
    grid: np.ndarray,
    agent_pos: Tuple[float, float],
    goal: Tuple[int, int],
    cell_size: int = 8,
    agent_color: Tuple[int, int, int] = (0, 180, 255),
    goal_color: Tuple[int, int, int] = (0, 170, 0),
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    rows, cols = grid.shape
    cell = max(2, int(cell_size))
    if background is None:
        grid_img = np.where(grid == 1, 0, 255).astype(np.uint8)
        img = Image.fromarray(grid_img, mode="L")
        resample = getattr(getattr(Image, "Resampling", Image), "NEAREST", None)
        if resample is None:
            resample = getattr(Image, "NEAREST", 0)
        img = img.resize((cols * cell, rows * cell), resample=resample).convert("RGB")
    else:
        img = Image.fromarray(background).convert("RGB")

    draw = ImageDraw.Draw(img)
    gr, gc = goal
    gx0, gy0 = gc * cell, gr * cell
    gx1, gy1 = gx0 + cell, gy0 + cell
    pad = max(2, cell // 6)
    draw.ellipse((gx0 + pad, gy0 + pad, gx1 - pad, gy1 - pad), fill=goal_color)

    ar, ac = agent_pos
    ax = (ac) * cell
    ay = (ar) * cell
    radius = max(3, int(cell * 0.45))
    draw.ellipse(
        (ax - radius, ay - radius, ax + radius, ay + radius),
        fill=agent_color,
        outline=(0, 0, 0),
        width=max(1, radius // 4),
    )
    return np.array(img)


def render_png_frame(
    png_path: str,
    grid: np.ndarray,
    agent_pos: Tuple[float, float],
    goal: Tuple[int, int],
    cell_size: int = 8,
    agent_color: Tuple[int, int, int] = (0, 180, 255),
    goal_color: Tuple[int, int, int] = (0, 170, 0),
) -> np.ndarray:
    from PIL import Image, ImageDraw

    base = Image.open(png_path).convert("RGBA")
    w, h = base.size
    cols = grid.shape[1]
    rows = grid.shape[0]
    draw = ImageDraw.Draw(base)

    gr, gc = goal
    gx = (gc + 0.5) * (w / cols)
    gy = (gr + 0.5) * (h / rows)
    radius = max(2, int(min(w / cols, h / rows) * 0.35))
    draw.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), fill=goal_color)

    ar, ac = agent_pos
    ax = (ac + 0.5) * (w / cols)
    ay = (ar + 0.5) * (h / rows)
    draw.ellipse((ax - radius, ay - radius, ax + radius, ay + radius), fill=agent_color)
    return np.array(base.convert("RGB"))


def _auto_pick_start_goal(
    grid: np.ndarray,
    start: Optional[Tuple[int, int]],
    goal: Optional[Tuple[int, int]],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    rows, cols = grid.shape
    open_cells = np.argwhere(grid == 0)
    if open_cells.size == 0:
        raise ValueError("Grid has no open cells to place start/goal.")

    border_opens: List[Tuple[int, int]] = []
    for r, c in open_cells:
        rr, cc = int(r), int(c)
        if rr == 0 or cc == 0 or rr == rows - 1 or cc == cols - 1:
            border_opens.append((rr, cc))

    if start is None:
        if border_opens:
            sr, sc = min(border_opens, key=lambda x: (x[0] + x[1], x[0], x[1]))
            start = (sr, sc)
        else:
            sr, sc = open_cells[0]
            start = (int(sr), int(sc))
    else:
        sr, sc = start

    if goal is None:
        if border_opens:
            gr, gc = max(
                border_opens,
                key=lambda x: (x[0] + x[1], x[0], x[1]),
            )
            goal = (gr, gc)
        else:
            dist = np.full((rows, cols), -1, dtype=np.int32)
            dist[sr, sc] = 0
            queue = [(sr, sc)]
            qi = 0
            while qi < len(queue):
                r, c = queue[qi]
                qi += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if grid[rr, cc] == 0 and dist[rr, cc] < 0:
                            dist[rr, cc] = dist[r, c] + 1
                            queue.append((rr, cc))

            max_dist = dist.max()
            if max_dist < 0:
                raise ValueError("Start is isolated; no reachable goal cells.")
            gr, gc = np.argwhere(dist == max_dist)[0]
            goal = (int(gr), int(gc))

    return start, goal


def find_latest_maze_id(maze_dir: str) -> int:
    pattern = re.compile(r"^maze_(\d{3})_meta\.json$")
    ids: List[int] = []
    for root, _, files in os.walk(maze_dir):
        for name in files:
            match = pattern.match(name)
            if match:
                ids.append(int(match.group(1)))

    if not ids:
        raise FileNotFoundError(
            f"No maze meta files found in {maze_dir}. Expected maze_###_meta.json"
        )

    return max(ids)


# ----------------------------
# Game UI
# ----------------------------

class MazeGame:
    def __init__(
        self,
        root: Any,
        maze_dir: str,
        maze_id: int,
        cell_size: int = 24,
        prefer_npy: bool = True,
        show_grid_lines: bool = True,
    ):
        self.root = root
        self.maze_dir = maze_dir
        self.maze_id = maze_id
        self.cell_size = max(6, int(cell_size))
        self.prefer_npy = prefer_npy
        self.show_grid_lines = show_grid_lines

        # Colors
        self.color_wall = "#000000"
        self.color_open = "#FFFFFF"
        self.color_agent = "#FF0000"
        self.color_goal = "#00AA00"
        self.color_gridline = "#DDDDDD"

        self.grid: Optional[np.ndarray] = None
        self.meta: Optional[Dict[str, Any]] = None
        self.rows = 0
        self.cols = 0
        self.start = (0, 0)
        self.goal = (0, 0)
        self.agent = (0, 0)
        self.agent_pos: Tuple[float, float] = (0.0, 0.0)
        self.keys_down: Set[str] = set()
        self.speed_cells_per_sec: float = 6.0
        self.agent_radius_cells: float = 0.35
        self._last_tick: float = time.perf_counter()
        self.moves = 0
        self.won = False

        self.status_var = tk.StringVar(value="")

        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        status = tk.Label(root, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self._bind_keys()
        self.load_maze(self.maze_id)
        self._schedule_tick()

    def _bind_keys(self):
        # Movement (continuous)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

        # Utility
        self.root.bind("r", lambda e: self.reset())
        self.root.bind("n", lambda e: self.load_maze(self.maze_id + 1))
        self.root.bind("p", lambda e: self.load_maze(self.maze_id - 1))
        self.root.bind("<Escape>", lambda e: self.root.quit())

        # Redraw on resize
        self.root.bind("<Configure>", lambda e: self.redraw())

    def _on_key_press(self, event):
        self.keys_down.add(event.keysym.lower())

    def _on_key_release(self, event):
        self.keys_down.discard(event.keysym.lower())

    def load_maze(self, maze_id: int):
        if maze_id < 0:
            return

        try:
            grid, meta = load_maze_files(
                self.maze_dir, maze_id, prefer_npy=self.prefer_npy
            )
        except Exception as ex:
            messagebox.showerror(
                "Load Maze Failed",
                f"Could not load maze {maze_id:03d}:\n\n{ex}",
            )
            return

        self.maze_id = maze_id
        self.grid = grid
        self.meta = meta
        self.rows, self.cols = grid.shape
        self.start = meta["start"]
        self.goal = meta["goal"]
        self.reset()

    def reset(self):
        self.agent = self.start
        self.agent_pos = (self.start[0] + 0.5, self.start[1] + 0.5)
        self.moves = 0
        self.won = False
        self._update_status()
        self._set_min_window()
        self.redraw()

    def _set_min_window(self):
        # Minimum size; user can resize bigger.
        w = self.cols * self.cell_size
        h = self.rows * self.cell_size + 30
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        max_w = max(360, screen_w - 80)
        max_h = max(360, screen_h - 120)
        self.root.minsize(min(max_w, max(360, w)), min(max_h, max(360, h)))

    def _schedule_tick(self):
        self.root.after(16, self._tick)

    def _tick(self):
        if self.grid is not None:
            now = time.perf_counter()
            dt = now - self._last_tick
            self._last_tick = now
            self._apply_movement(dt)
        self._schedule_tick()

    def _apply_movement(self, dt: float):
        if self.won or dt <= 0:
            return

        dr = 0.0
        dc = 0.0
        if "up" in self.keys_down or "w" in self.keys_down:
            dr -= 1
        if "down" in self.keys_down or "s" in self.keys_down:
            dr += 1
        if "left" in self.keys_down or "a" in self.keys_down:
            dc -= 1
        if "right" in self.keys_down or "d" in self.keys_down:
            dc += 1

        if dr == 0 and dc == 0:
            return

        mag = math.hypot(dr, dc)
        dr /= mag
        dc /= mag

        step = self.speed_cells_per_sec * dt
        prev_cell = self.agent
        nr = self.agent_pos[0] + dr * step
        nc = self.agent_pos[1] + dc * step

        # axis-separated movement to slide along walls
        r_try = (float(nr), float(self.agent_pos[1]))
        if self._is_open_at(r_try[0], r_try[1]):
            self.agent_pos = r_try

        c_try = (float(self.agent_pos[0]), float(nc))
        if self._is_open_at(c_try[0], c_try[1]):
            self.agent_pos = c_try

        ar, ac = self.agent_pos
        self.agent = (int(ar), int(ac))
        if self.agent != prev_cell:
            self.moves += 1

        gr, gc = self.goal
        goal_center = (gr + 0.5, gc + 0.5)
        if float((ar - goal_center[0]) ** 2 + (ac - goal_center[1]) ** 2) <= (
            self.agent_radius_cells ** 2
        ):
            self.won = True
            self._update_status()
            self.redraw()
            messagebox.showinfo("You win", f"Goal reached in {self.moves} moves.")
            return

        self._update_status()
        self.redraw()

    def _is_open_at(self, r: float, c: float) -> bool:
        if self.grid is None:
            return False
        if r < 0 or c < 0 or r >= self.rows or c >= self.cols:
            return False

        radius = self.agent_radius_cells
        for rr in (r - radius, r, r + radius):
            for cc in (c - radius, c, c + radius):
                gr = int(rr)
                gc = int(cc)
                if gr < 0 or gc < 0 or gr >= self.rows or gc >= self.cols:
                    return False
                if self.grid[gr, gc] == 1:
                    return False
        return True

    def _update_status(self):
        sr, sc = self.start
        gr, gc = self.goal
        ar, ac = self.agent
        win_text = " | WIN" if self.won else ""
        self.status_var.set(
            f"Maze {self.maze_id:03d} | Moves: {self.moves} | "
            f"Start: ({sr},{sc}) | Goal: ({gr},{gc}) | Agent: ({ar},{ac}){win_text} | "
            "Controls: Arrows/WASD move, R reset, N next, P prev, Esc quit"
        )

    def redraw(self):
        if self.grid is None:
            return

        self.canvas.delete("all")

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        # Scale down to fit the window if needed.
        cs = min(
            self.cell_size,
            max(4, min(canvas_w // self.cols, canvas_h // self.rows)),
        )

        maze_w = self.cols * cs
        maze_h = self.rows * cs

        offset_x = max(0, (canvas_w - maze_w) // 2)
        offset_y = max(0, (canvas_h - maze_h) // 2)

        outline = ""

        # draw cells
        for r in range(self.rows):
            y0 = offset_y + r * cs
            y1 = y0 + cs
            for c in range(self.cols):
                x0 = offset_x + c * cs
                x1 = x0 + cs
                fill = self.color_wall if self.grid[r, c] == 1 else self.color_open
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline)

        # goal
        gr, gc = self.goal
        gx0 = offset_x + gc * cs
        gy0 = offset_y + gr * cs
        gx1 = gx0 + cs
        gy1 = gy0 + cs
        pad = max(2, cs // 6)
        self.canvas.create_oval(
            gx0 + pad, gy0 + pad, gx1 - pad, gy1 - pad,
            fill=self.color_goal, outline=""
        )

        # agent
        ar, ac = self.agent_pos
        ax0 = float(offset_x + ac * cs)
        ay0 = float(offset_y + ar * cs)
        ax1 = float(ax0 + cs)
        ay1 = float(ay0 + cs)
        self.canvas.create_oval(
            ax0 + pad, ay0 + pad, ax1 - pad, ay1 - pad,
            fill=self.color_agent, outline=""
        )


def main():
    parser = argparse.ArgumentParser(description="Playable Tkinter Maze Game")
    parser.add_argument("--maze_dir", type=str, default="data/mazes",
                        help="Folder containing maze_###_*.{npy,txt,json}")
    parser.add_argument("--maze_id", type=int, default=None,
                        help="Maze number, e.g., 1 => maze_001_* (default: latest)")
    parser.add_argument("--cell_size", type=int, default=24,
                        help="Pixels per cell")
    parser.add_argument("--prefer_npy", action="store_true",
                        help="Prefer .npy grid if available")
    parser.add_argument("--no_grid_lines", action="store_true",
                        help="Disable cell grid lines")
    args = parser.parse_args()

    if tk is None or messagebox is None:
        raise RuntimeError("tkinter is not available; install it to run the GUI.")
    root = tk.Tk()
    root.title("Maze Game")

    maze_id = args.maze_id
    if maze_id is None:
        maze_id = find_latest_maze_id(args.maze_dir)

    MazeGame(
        root=root,
        maze_dir=args.maze_dir,
        maze_id=maze_id,
        cell_size=args.cell_size,
        prefer_npy=args.prefer_npy,
        show_grid_lines=(not args.no_grid_lines),
    )

    root.mainloop()


if __name__ == "__main__":
    main()
