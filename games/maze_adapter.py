from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from games.base import GameAdapter


def _latest_meta_path(root: Path) -> Path:
    meta_files = list(root.glob("*/**/*_meta.json"))
    if not meta_files:
        raise FileNotFoundError("No maze meta files found under data/mazes.")
    return max(meta_files, key=lambda p: p.stat().st_mtime)


def _resolve_maze_paths() -> Dict[str, Path]:
    base = Path("data/mazes")
    maze_id = os.getenv("MAZE_ID", "").strip()
    if maze_id:
        meta_path = base / maze_id / f"{maze_id}_meta.json"
    else:
        meta_path = _latest_meta_path(base)
        maze_id = meta_path.stem.replace("_meta", "")

    grid_path = meta_path.with_name(f"{maze_id}_grid.npy")
    grid_xo_path = meta_path.with_name(f"{maze_id}_grid_xo.txt")
    if not grid_path.exists() and not grid_xo_path.exists():
        raise FileNotFoundError(f"Missing grid file: {grid_path} (or {grid_xo_path})")

    return {"maze_id": maze_id, "meta": meta_path, "grid": grid_path, "grid_xo": grid_xo_path}


def _load_grid(grid_path: Path, grid_xo_path: Path) -> np.ndarray:
    if grid_path.exists():
        return np.load(grid_path)
    text = grid_xo_path.read_text(encoding="utf-8").strip().splitlines()
    rows = len(text)
    cols = max(len(line) for line in text) if rows else 0
    grid = np.ones((rows, cols), dtype=np.uint8)
    for r, line in enumerate(text):
        for c, ch in enumerate(line):
            if ch in ("O", "S", "G"):
                grid[r, c] = 0
            elif ch == "X":
                grid[r, c] = 1
    return grid


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        variant: Optional[int] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self._variant = variant

        paths = _resolve_maze_paths()
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        self._meta = meta
        self._maze_id = paths["maze_id"]
        self._grid = _load_grid(paths["grid"], paths["grid_xo"])
        self._rows, self._cols = self._grid.shape
        max_steps_env = os.getenv("MAZE_MAX_STEPS", "").strip()
        if max_steps_env:
            try:
                self._max_steps = max(1, int(max_steps_env))
            except ValueError:
                self._max_steps = int(self._rows * self._cols)
        else:
            self._max_steps = int(self._rows * self._cols)
        self._step_count = 0
        self._wall_penalty = -5.0
        self._step_penalty = -0.01
        self._goal_bonus = 10.0

        start = meta.get("start")
        goal = meta.get("goal")
        self._start = self._sanitize_point(start, fallback="start")
        self._goal = self._sanitize_point(goal, fallback="goal")
        self._dist_map = self._compute_distances(self._goal)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self._bg = None
        self._cell_size = 10
        self._png_path = Path(meta.get("png", ""))
        if self._png_path and not self._png_path.is_absolute():
            self._png_path = (Path.cwd() / self._png_path).resolve()

    def _sanitize_point(self, value: Optional[list], fallback: str) -> tuple[int, int]:
        if value and len(value) == 2:
            r, c = int(value[0]), int(value[1])
            if 0 <= r < self._rows and 0 <= c < self._cols and self._grid[r, c] == 0:
                return r, c
        if fallback == "start":
            opens = np.argwhere(self._grid == 0)
            if opens.size == 0:
                raise ValueError("Maze has no open cells.")
            return tuple(opens[0])
        opens = np.argwhere(self._grid == 0)
        if opens.size == 0:
            raise ValueError("Maze has no open cells.")
        return tuple(opens[-1])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent = tuple(self._start)
        self._step_count = 0
        self._prev_dist = self._dist_at(self._agent)
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(action, (0, 0))
        r, c = self._agent
        nr, nc = r + dr, c + dc

        reward = self._step_penalty
        if 0 <= nr < self._rows and 0 <= nc < self._cols and self._grid[nr, nc] == 0:
            self._agent = (nr, nc)
            new_dist = self._dist_at(self._agent)
            if np.isfinite(self._prev_dist) and np.isfinite(new_dist):
                reward += 0.1 * (self._prev_dist - new_dist)
            self._prev_dist = new_dist
        else:
            reward += self._wall_penalty

        self._step_count += 1
        terminated = self._agent == self._goal
        if terminated:
            reward += self._goal_bonus
        truncated = self._step_count >= self._max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _dist_at(self, pos: tuple[int, int]) -> float:
        r, c = pos
        return float(self._dist_map[r, c])

    def _compute_distances(self, goal: tuple[int, int]) -> np.ndarray:
        dist = np.full((self._rows, self._cols), np.inf, dtype=np.float32)
        gr, gc = goal
        if self._grid[gr, gc] != 0:
            return dist
        dist[gr, gc] = 0.0
        queue = [(gr, gc)]
        for r, c in queue:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self._rows and 0 <= nc < self._cols and self._grid[nr, nc] == 0:
                    if dist[nr, nc] == np.inf:
                        dist[nr, nc] = dist[r, c] + 1.0
                        queue.append((nr, nc))
        return dist

    def _obs(self) -> np.ndarray:
        ar, ac = self._agent
        gr, gc = self._goal
        return np.array(
            [
                ar / max(1, self._rows - 1),
                ac / max(1, self._cols - 1),
                gr / max(1, self._rows - 1),
                gc / max(1, self._cols - 1),
            ],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._bg is None:
            cell = self._cell_size
            w, h = self._cols * cell, self._rows * cell
            base = Image.new("RGB", (w, h), (255, 255, 255))
            draw = ImageDraw.Draw(base)
            for r in range(self._rows):
                for c in range(self._cols):
                    if self._grid[r, c] == 1:
                        x0, y0 = c * cell, r * cell
                        x1, y1 = x0 + cell, y0 + cell
                        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
            self._bg = base

        frame = self._bg.copy()
        draw = ImageDraw.Draw(frame)
        w, h = frame.size
        x = (self._agent[1] + 0.5) * (w / self._cols)
        y = (self._agent[0] + 0.5) * (h / self._rows)
        radius = max(2, int(self._cell_size * 0.35))
        palette = [
            (0, 180, 255),
            (255, 120, 0),
            (120, 220, 0),
            (180, 0, 255),
            (255, 0, 90),
            (0, 220, 180),
            (255, 200, 0),
            (60, 120, 255),
        ]
        color = palette[0]
        if self._variant is not None:
            color = palette[int(self._variant) % len(palette)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        return np.array(frame)


def _make_env(render_mode: Optional[str], seed: Optional[int], variant: Optional[int]) -> gym.Env:
    env = MazeEnv(render_mode=render_mode, seed=seed, variant=variant)
    if seed is not None:
        env.reset(seed=seed)
    return env


def maze_adapter() -> GameAdapter:
    return GameAdapter(
        name="maze",
        description="Grid-based maze environment backed by a maze PNG and cached grid.",
        model_prefix="ppo_maze",
        make_env_fn=_make_env,
        extra_metrics=[],
        eval_fn=None,
        heatmap_fn=None,
    )
