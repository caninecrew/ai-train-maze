from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

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
        self._wall_penalty = -0.5
        self._consec_wall_penalty = -0.25
        self._step_penalty = -0.0005
        self._goal_bonus = 500.0
        self._idle_penalty = -0.001
        self._move_bonus = 0.01
        self._cookie_bonus = 0.05
        self._shaping_coef = 0.1
        self._novelty_bonus = 0.02
        self._backtrack_penalty = -0.1
        self._terminate_on_wall = os.getenv("MAZE_TERMINATE_ON_WALL", "").strip().lower() in {"1", "true", "yes", "on"}
        self._best_dist_bonus = 0.1
        self._best_dist_hit_bonus = 0.2
        self._best_progress_bonus = 0.08
        self._helper_count = 0
        self._helper_base_bonus = 0.25
        self._helper_power = 1.5
        self._helper_decay_steps = 0

        start = self._meta.get("start")
        goal = self._meta.get("goal")
        self._start = self._sanitize_point(start, fallback="start")
        base_goal = self._sanitize_point(goal, fallback="goal")
        self._goal = self._apply_training_goal(base_goal)
        self._dist_map = self._compute_distances(self._goal)
        self._helper_cells: List[tuple[int, int]] = []
        self._helper_weights: List[float] = []

        self._obs_mode = os.getenv("MAZE_OBS_MODE", "pos").strip().lower()
        sensor_range_env = os.getenv("MAZE_SENSOR_RANGE", "").strip()
        self._sensor_range = None
        if sensor_range_env:
            try:
                self._sensor_range = max(1, int(sensor_range_env))
            except ValueError:
                self._sensor_range = None

        self.action_space = spaces.Discrete(4)
        obs_size = 4
        if self._obs_mode in {"rays", "pos+rays"}:
            obs_size = 8 if self._obs_mode == "rays" else 12
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        cell_size_env = os.getenv("MAZE_CELL_SIZE", "").strip()
        if cell_size_env:
            try:
                self._cell_size = max(2, int(cell_size_env))
            except ValueError:
                self._cell_size = 8
        else:
            self._cell_size = 8
        move_bonus_env = os.getenv("MAZE_MOVE_BONUS", "").strip()
        if move_bonus_env:
            try:
                self._move_bonus = float(move_bonus_env)
            except ValueError:
                self._move_bonus = 0.01
        cookie_bonus_env = os.getenv("MAZE_COOKIE_BONUS", "").strip()
        if cookie_bonus_env:
            try:
                self._cookie_bonus = float(cookie_bonus_env)
            except ValueError:
                self._cookie_bonus = 0.05
        shaping_coef_env = os.getenv("MAZE_SHAPING_COEF", "").strip()
        if shaping_coef_env:
            try:
                self._shaping_coef = float(shaping_coef_env)
            except ValueError:
                self._shaping_coef = 0.1
        novelty_bonus_env = os.getenv("MAZE_NOVELTY_BONUS", "").strip()
        if novelty_bonus_env:
            try:
                self._novelty_bonus = float(novelty_bonus_env)
            except ValueError:
                self._novelty_bonus = 0.02
        backtrack_penalty_env = os.getenv("MAZE_BACKTRACK_PENALTY", "").strip()
        if backtrack_penalty_env:
            try:
                self._backtrack_penalty = float(backtrack_penalty_env)
            except ValueError:
                self._backtrack_penalty = -0.2
        wall_penalty_env = os.getenv("MAZE_WALL_PENALTY", "").strip()
        if wall_penalty_env:
            try:
                self._wall_penalty = float(wall_penalty_env)
            except ValueError:
                self._wall_penalty = -0.5
        consec_wall_penalty_env = os.getenv("MAZE_CONSEC_WALL_PENALTY", "").strip()
        if consec_wall_penalty_env:
            try:
                self._consec_wall_penalty = float(consec_wall_penalty_env)
            except ValueError:
                self._consec_wall_penalty = -0.25
        helper_count_env = os.getenv("MAZE_HELPER_DOTS", "").strip()
        if helper_count_env:
            try:
                self._helper_count = max(0, int(helper_count_env))
            except ValueError:
                self._helper_count = 0
        helper_bonus_env = os.getenv("MAZE_HELPER_BONUS", "").strip()
        if helper_bonus_env:
            try:
                self._helper_base_bonus = float(helper_bonus_env)
            except ValueError:
                self._helper_base_bonus = 0.25
        helper_power_env = os.getenv("MAZE_HELPER_POWER", "").strip()
        if helper_power_env:
            try:
                self._helper_power = float(helper_power_env)
            except ValueError:
                self._helper_power = 1.5
        helper_decay_env = os.getenv("MAZE_HELPER_DECAY_STEPS", "").strip()
        if helper_decay_env:
            try:
                self._helper_decay_steps = max(0, int(helper_decay_env))
            except ValueError:
                self._helper_decay_steps = 0
        if self._helper_count > 0 and self._helper_decay_steps == 0:
            self._helper_decay_steps = max(1, int(self._max_steps * 0.6))
        if self._helper_count > 0:
            self._helper_cells, self._helper_weights = self._build_helpers()

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
            fraction = float(fraction_env) if fraction_env else 0.1
        except ValueError:
            fraction = 0.1
        fraction = max(0.1, min(0.9, fraction))
        target_dist = max(1, int(goal_dist * fraction))
        candidates = np.argwhere(dist_start == target_dist)
        if candidates.size == 0:
            return base_goal
        return tuple(candidates[0])

    def _build_helpers(self) -> tuple[List[tuple[int, int]], List[float]]:
        dist_start = self._compute_distances(self._start)
        goal_dist = float(dist_start[self._goal])
        if not np.isfinite(goal_dist) or goal_dist < 2:
            return [], []
        on_path = np.isfinite(dist_start) & np.isfinite(self._dist_map)
        on_path &= (dist_start + self._dist_map) == goal_dist
        targets = []
        for idx in range(1, self._helper_count + 1):
            frac = idx / float(self._helper_count + 1)
            target = int(round(goal_dist * frac))
            target = max(1, min(int(goal_dist) - 1, target))
            if target not in targets:
                targets.append(target)
        helper_cells: List[tuple[int, int]] = []
        helper_weights: List[float] = []
        for target in targets:
            candidates = np.argwhere(on_path & (dist_start == target))
            if candidates.size == 0:
                path_cells = np.argwhere(on_path)
                if path_cells.size == 0:
                    continue
                dist_vals = dist_start[path_cells[:, 0], path_cells[:, 1]]
                idx = int(np.argmin(np.abs(dist_vals - target)))
                pick = tuple(path_cells[idx])
            else:
                pick = tuple(candidates[0])
            if pick in helper_cells or pick == self._start or pick == self._goal:
                continue
            helper_cells.append(pick)
            progress = float(target) / goal_dist
            weight = progress ** max(0.1, self._helper_power)
            helper_weights.append(weight)
        return helper_cells, helper_weights

    def _helper_decay_factor(self) -> float:
        if self._helper_decay_steps <= 0:
            return 1.0
        remaining = max(0.0, self._helper_decay_steps - self._step_count)
        return remaining / float(self._helper_decay_steps)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent = tuple(self._start)
        self._step_count = 0
        self._prev_dist = self._dist_at(self._agent)
        self._best_dist = self._prev_dist
        self._best_dist_hits = 0
        self._start_dist = self._prev_dist
        self._visited = {self._agent}
        self._wall_hits = 0
        self._consec_wall_hits = 0
        self._idle_steps = 0
        self._backtracks = 0
        self._novel_steps = 0
        self._last_pos = self._agent
        self._prev_action = None
        self._helper_hits = set()
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(action, (0, 0))
        r, c = self._agent
        nr, nc = r + dr, c + dc

        reward = self._step_penalty
        terminated = False
        if 0 <= nr < self._rows and 0 <= nc < self._cols and self._grid[nr, nc] == 0:
            self._agent = (nr, nc)
            self._consec_wall_hits = 0
            reward += self._move_bonus
            new_dist = self._dist_at(self._agent)
            if np.isfinite(self._prev_dist) and np.isfinite(new_dist):
                reward += self._shaping_coef * (self._prev_dist - new_dist)
                if new_dist < self._prev_dist:
                    reward += self._cookie_bonus
            if np.isfinite(new_dist) and new_dist < self._best_dist:
                reward += self._best_dist_bonus * (self._best_dist - new_dist)
                self._best_dist = new_dist
                self._best_dist_hits += 1
                reward += self._best_dist_hit_bonus * self._best_dist_hits
            self._prev_dist = new_dist
            if self._agent not in self._visited:
                reward += self._novelty_bonus
                self._visited.add(self._agent)
                self._novel_steps += 1
            if self._helper_cells:
                try:
                    idx = self._helper_cells.index(self._agent)
                except ValueError:
                    idx = -1
                if idx >= 0 and idx not in self._helper_hits:
                    decay = self._helper_decay_factor()
                    reward += self._helper_base_bonus * self._helper_weights[idx] * decay
                    self._helper_hits.add(idx)
            else:
                reward += -0.01
            reverse_map = {0: 1, 1: 0, 2: 3, 3: 2}
            if self._prev_action is not None and action == reverse_map.get(self._prev_action):
                reward += self._backtrack_penalty
                self._backtracks += 1
        else:
            reward += self._wall_penalty
            self._wall_hits += 1
            self._consec_wall_hits += 1
            if self._consec_wall_hits > 1:
                reward += self._consec_wall_penalty * (self._consec_wall_hits - 1)
            if self._terminate_on_wall:
                terminated = True
            else:
                terminated = False
        if self._agent == (r, c):
            reward += self._idle_penalty
            self._idle_steps += 1
        self._last_pos = (r, c)
        self._prev_action = action

        self._step_count += 1
        if self._agent == self._goal:
            terminated = True
        if terminated:
            reward += self._goal_bonus
        truncated = self._step_count >= self._max_steps
        if terminated or truncated:
            if np.isfinite(self._start_dist) and np.isfinite(self._best_dist):
                reward += self._best_progress_bonus * (self._start_dist - self._best_dist)
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
        base = [
            ar / max(1, self._rows - 1),
            ac / max(1, self._cols - 1),
            gr / max(1, self._rows - 1),
            gc / max(1, self._cols - 1),
        ]
        if self._obs_mode == "pos":
            return np.array(base, dtype=np.float32)
        rays = self._ray_sensors()
        if self._obs_mode == "rays":
            return np.array(rays, dtype=np.float32)
        return np.array(base + rays, dtype=np.float32)

    def _ray_sensors(self) -> List[float]:
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        max_dist = self._sensor_range
        if max_dist is None:
            max_dist = max(self._rows, self._cols)
        distances = []
        for dr, dc in directions:
            dist = self._ray_distance(dr, dc, max_dist)
            distances.append(dist / max_dist)
        return distances

    def _ray_distance(self, dr: int, dc: int, max_dist: int) -> int:
        r, c = self._agent
        for step in range(1, max_dist + 1):
            nr, nc = r + dr * step, c + dc * step
            if not (0 <= nr < self._rows and 0 <= nc < self._cols):
                return step
            if self._grid[nr, nc] != 0:
                return step
        return max_dist

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

    def get_eval_stats(self) -> Dict[str, float]:
        r, c = self._agent
        gr, gc = self._goal
        return {
            "goal_reached": float(self._agent == self._goal),
            "best_dist": float(self._best_dist),
            "best_progress": float(self._start_dist - self._best_dist),
            "steps": float(self._step_count),
            "wall_hits": float(self._wall_hits),
            "idle_steps": float(self._idle_steps),
            "backtracks": float(self._backtracks),
            "novel_steps": float(self._novel_steps),
            "unique_cells": float(len(self._visited)),
            "final_dist": float(self._dist_at(self._agent)),
            "final_row": float(r),
            "final_col": float(c),
            "goal_row": float(gr),
            "goal_col": float(gc),
        }


def _evaluate_maze(model: Any, episodes: int, deterministic: bool = True) -> Dict[str, float]:
    seed_env = os.getenv("MAZE_EVAL_SEED_BASE", "").strip()
    try:
        seed_base = int(seed_env) if seed_env else None
    except ValueError:
        seed_base = None
    env = _make_env(render_mode=None, seed=None, variant=None)
    rewards: List[float] = []
    lengths: List[int] = []
    goals: List[float] = []
    best_dists: List[float] = []
    best_progress: List[float] = []
    steps: List[float] = []
    wall_hits: List[float] = []
    idle_steps: List[float] = []
    backtracks: List[float] = []
    novel_steps: List[float] = []
    unique_cells: List[float] = []
    final_dist: List[float] = []
    final_rows: List[float] = []
    final_cols: List[float] = []
    wall_hit_rate: List[float] = []
    idle_rate: List[float] = []
    backtrack_rate: List[float] = []
    novel_rate: List[float] = []
    for _ in range(episodes):
        if seed_base is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=seed_base + _)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(ep_steps)
        stats = env.get_eval_stats() if hasattr(env, "get_eval_stats") else {}
        goal_reached = float(stats.get("goal_reached", 0.0))
        if not goal_reached:
            best_dist_val = stats.get("best_dist")
            final_row = stats.get("final_row")
            final_col = stats.get("final_col")
            goal_row = stats.get("goal_row")
            goal_col = stats.get("goal_col")
            if best_dist_val is not None and float(best_dist_val) <= 0.0:
                goal_reached = 1.0
            elif (
                final_row is not None
                and final_col is not None
                and goal_row is not None
                and goal_col is not None
                and int(final_row) == int(goal_row)
                and int(final_col) == int(goal_col)
            ):
                goal_reached = 1.0
        goals.append(goal_reached)
        best_dists.append(float(stats.get("best_dist", 0.0)))
        best_progress.append(float(stats.get("best_progress", 0.0)))
        steps.append(float(stats.get("steps", ep_steps)))
        wall_hits.append(float(stats.get("wall_hits", 0.0)))
        idle_steps.append(float(stats.get("idle_steps", 0.0)))
        backtracks.append(float(stats.get("backtracks", 0.0)))
        novel_steps.append(float(stats.get("novel_steps", 0.0)))
        unique_cells.append(float(stats.get("unique_cells", 0.0)))
        final_dist.append(float(stats.get("final_dist", 0.0)))
        final_rows.append(float(stats.get("final_row", np.nan)))
        final_cols.append(float(stats.get("final_col", np.nan)))
        denom = float(stats.get("steps", ep_steps)) or 1.0
        wall_hit_rate.append(float(stats.get("wall_hits", 0.0)) / denom)
        idle_rate.append(float(stats.get("idle_steps", 0.0)) / denom)
        backtrack_rate.append(float(stats.get("backtracks", 0.0)) / denom)
        novel_rate.append(float(stats.get("novel_steps", 0.0)) / denom)
    env.close()

    def _ci(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        std = float(np.std(values, ddof=1))
        return 1.96 * std / np.sqrt(len(values))

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    return {
        "avg_reward": avg_reward,
        "avg_reward_ci": _ci(rewards),
        "avg_ep_len": avg_len,
        "avg_ep_len_ci": _ci(lengths),
        "goal_reached_rate": float(np.mean(goals)) if goals else 0.0,
        "best_dist": float(np.mean(best_dists)) if best_dists else 0.0,
        "best_progress": float(np.mean(best_progress)) if best_progress else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "wall_hits": float(np.mean(wall_hits)) if wall_hits else 0.0,
        "idle_steps": float(np.mean(idle_steps)) if idle_steps else 0.0,
        "backtracks": float(np.mean(backtracks)) if backtracks else 0.0,
        "novel_steps": float(np.mean(novel_steps)) if novel_steps else 0.0,
        "unique_cells": float(np.mean(unique_cells)) if unique_cells else 0.0,
        "final_dist": float(np.mean(final_dist)) if final_dist else 0.0,
        "final_row": float(np.nanmean(final_rows)) if final_rows else float("nan"),
        "final_col": float(np.nanmean(final_cols)) if final_cols else float("nan"),
        "wall_hit_rate": float(np.mean(wall_hit_rate)) if wall_hit_rate else 0.0,
        "idle_rate": float(np.mean(idle_rate)) if idle_rate else 0.0,
        "backtrack_rate": float(np.mean(backtrack_rate)) if backtrack_rate else 0.0,
        "novel_rate": float(np.mean(novel_rate)) if novel_rate else 0.0,
    }
    def get_eval_stats(self) -> Dict[str, float]:
        return {
            "goal_reached": float(self._agent == self._goal),
            "best_dist": float(self._best_dist),
            "best_progress": float(self._start_dist - self._best_dist),
            "steps": float(self._step_count),
        }


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
        extra_metrics=[
            "goal_reached_rate",
            "best_dist",
            "best_progress",
            "avg_steps",
            "wall_hits",
            "idle_steps",
            "backtracks",
            "novel_steps",
            "unique_cells",
        "final_dist",
        "final_row",
        "final_col",
        "goal_row",
        "goal_col",
        "wall_hit_rate",
        "idle_rate",
        "backtrack_rate",
        "novel_rate",
        ],
        eval_fn=_evaluate_maze,
        heatmap_fn=None,
    )
