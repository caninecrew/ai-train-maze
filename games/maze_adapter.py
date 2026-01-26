from __future__ import annotations

import os
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from maze_game import find_latest_maze_id, load_maze_files, render_grid_frame

from games.base import GameAdapter


def _resolve_maze_id(maze_dir: str) -> str:
    maze_id = os.getenv("MAZE_ID", "").strip()
    if maze_id:
        raw = str(maze_id).strip()
        if raw.lower().startswith("maze_"):
            raw = raw.split("maze_", 1)[1]
        return raw.zfill(3)
    return str(find_latest_maze_id(maze_dir)).zfill(3)


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

        maze_dir = os.getenv("MAZE_DIR", "data/mazes")
        maze_id = _resolve_maze_id(maze_dir)
        self._grid, self._meta = load_maze_files(maze_dir, maze_id, prefer_npy=True)
        self._maze_id = maze_id
        self._rows, self._cols = self._grid.shape
        max_steps_env = os.getenv("MAZE_MAX_STEPS", "").strip()
        if max_steps_env:
            try:
                self._max_steps = max(1, int(max_steps_env))
            except ValueError:
                self._max_steps = int(self._rows * self._cols * 0.75)
        else:
            self._max_steps = int(self._rows * self._cols * 0.75)
        self._step_count = 0
        self._wall_penalty = -2.0
        self._step_penalty = -0.005
        self._goal_bonus = 100.0
        self._idle_penalty = -0.05
        self._shaping_coef = 0.5
        self._novelty_bonus = 0.05
        self._backtrack_penalty = -0.1
        self._best_dist_bonus = 0.2

        start = self._meta.get("start")
        goal = self._meta.get("goal")
        self._start = self._sanitize_point(start, fallback="start")
        base_goal = self._sanitize_point(goal, fallback="goal")
        self._goal = self._apply_training_goal(base_goal)
        self._dist_map = self._compute_distances(self._goal)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        cell_size_env = os.getenv("MAZE_CELL_SIZE", "").strip()
        if cell_size_env:
            try:
                self._cell_size = max(2, int(cell_size_env))
            except ValueError:
                self._cell_size = 8
        else:
            self._cell_size = 8

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

    def _apply_training_goal(self, base_goal: tuple[int, int]) -> tuple[int, int]:
        train_mode = os.getenv("MAZE_TRAIN_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
        if not train_mode:
            return base_goal
        raw_goal = os.getenv("MAZE_TRAIN_GOAL", "").strip()
        if raw_goal:
            parts = raw_goal.replace(" ", "").split(",")
            if len(parts) == 2:
                try:
                    r, c = int(parts[0]), int(parts[1])
                    if 0 <= r < self._rows and 0 <= c < self._cols and self._grid[r, c] == 0:
                        return (r, c)
                except ValueError:
                    pass
        dist_start = self._compute_distances(self._start)
        goal_dist = dist_start[base_goal]
        if not np.isfinite(goal_dist) or goal_dist < 2:
            return base_goal
        fraction_env = os.getenv("MAZE_TRAIN_GOAL_FRACTION", "").strip()
        try:
            fraction = float(fraction_env) if fraction_env else 0.35
        except ValueError:
            fraction = 0.35
        fraction = max(0.1, min(0.9, fraction))
        target_dist = max(1, int(goal_dist * fraction))
        candidates = np.argwhere(dist_start == target_dist)
        if candidates.size == 0:
            return base_goal
        return tuple(candidates[0])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent = tuple(self._start)
        self._step_count = 0
        self._prev_dist = self._dist_at(self._agent)
        self._best_dist = self._prev_dist
        self._visited = {self._agent}
        self._last_pos = self._agent
        self._prev_action = None
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
                reward += self._shaping_coef * (self._prev_dist - new_dist)
            if np.isfinite(new_dist) and new_dist < self._best_dist:
                reward += self._best_dist_bonus * (self._best_dist - new_dist)
                self._best_dist = new_dist
            self._prev_dist = new_dist
            if self._agent not in self._visited:
                reward += self._novelty_bonus
                self._visited.add(self._agent)
            reverse_map = {0: 1, 1: 0, 2: 3, 3: 2}
            if self._prev_action is not None and action == reverse_map.get(self._prev_action):
                reward += self._backtrack_penalty
        else:
            reward += self._wall_penalty
        if self._agent == (r, c):
            reward += self._idle_penalty
        self._last_pos = (r, c)
        self._prev_action = action

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
        return render_grid_frame(
            self._grid,
            agent_pos=(self._agent[0] + 0.5, self._agent[1] + 0.5),
            goal=self._goal,
            cell_size=self._cell_size,
            agent_color=color,
        )

    def get_agent_cell(self) -> tuple[int, int]:
        return self._agent


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
