from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from maze_game import find_latest_maze_id, load_maze_files


@dataclass
class EvoConfig:
    maze_dir: str = "data/mazes"
    maze_id: str = ""
    cycles: int = 20
    population: int = 50
    max_steps: int = 2400
    min_alive: int = 5
    sensor_range: int = 20
    top_k: int = 10
    mutation_base: float = 0.05
    mutation_age: float = 0.2
    move_start_on_success: bool = True
    batch_size: int = 100
    epochs: int = 15
    learning_rate: float = 1e-3
    model_path: str = "models/evo_maze.pt"
    log_dir: str = "logs"


def _resolve_maze_id(maze_dir: str, raw_id: str) -> str:
    if raw_id:
        raw = str(raw_id).strip()
        if raw.lower().startswith("maze_"):
            raw = raw.split("maze_", 1)[1]
        return raw.zfill(3)
    return str(find_latest_maze_id(maze_dir)).zfill(3)


def _compute_distances(grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    rows, cols = grid.shape
    dist = np.full((rows, cols), np.inf, dtype=np.float32)
    gr, gc = goal
    if grid[gr, gc] != 0:
        return dist
    dist[gr, gc] = 0.0
    queue: List[Tuple[int, int]] = [(gr, gc)]
    for r, c in queue:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                if dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1.0
                    queue.append((nr, nc))
    return dist


def _ray_distance(grid: np.ndarray, pos: Tuple[int, int], dr: int, dc: int, max_dist: int) -> int:
    rows, cols = grid.shape
    r, c = pos
    for step in range(1, max_dist + 1):
        nr, nc = r + dr * step, c + dc * step
        if not (0 <= nr < rows and 0 <= nc < cols):
            return step
        if grid[nr, nc] != 0:
            return step
    return max_dist


def _ray_sensors(grid: np.ndarray, pos: Tuple[int, int], max_dist: int) -> List[float]:
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    distances = []
    for dr, dc in directions:
        dist = _ray_distance(grid, pos, dr, dc, max_dist)
        distances.append(dist / max_dist)
    return distances


class EvoModel(nn.Module):
    def __init__(self, input_size: int = 8, hidden: int = 32, output: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent:
    def __init__(self, start: Tuple[int, int]) -> None:
        self.pos = start
        self.alive = True
        self.steps = 0
        self.trajectory: List[Tuple[List[float], int]] = []
        self.prev_action: int | None = None


def _choose_action(
    model: EvoModel | None,
    obs: List[float],
    mutation_prob: float,
    prev_action: int | None,
) -> int:
    if model is None or random.random() < mutation_prob:
        return random.randint(0, 3)
    with torch.no_grad():
        logits = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        ranked = torch.argsort(logits, dim=1, descending=True).squeeze(0).tolist()
        if prev_action is None:
            return int(ranked[0])
        reverse_map = {0: 1, 1: 0, 2: 3, 3: 2}
        reverse = reverse_map.get(prev_action)
        for action in ranked:
            if action != reverse:
                return int(action)
        return int(ranked[0])


def _step_agent(grid: np.ndarray, agent: Agent, action: int) -> bool:
    dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(action, (0, 0))
    r, c = agent.pos
    nr, nc = r + dr, c + dc
    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
        agent.pos = (nr, nc)
        return True
    agent.alive = False
    return False


def _train_model(
    model: EvoModel,
    data: List[Tuple[List[float], int]],
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> float:
    if not data:
        return 0.0
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:] if split > 0 else data
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float("inf")
    patience = 2
    no_improve = 0

    for _ in range(epochs):
        random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            x = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            y = torch.tensor([b[1] for b in batch], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        if not val_data:
            continue
        with torch.no_grad():
            x_val = torch.tensor([b[0] for b in val_data], dtype=torch.float32)
            y_val = torch.tensor([b[1] for b in val_data], dtype=torch.long)
            val_loss = float(loss_fn(model(x_val), y_val).item())
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_val if best_val != float("inf") else 0.0


def _random_open_cell(grid: np.ndarray) -> Tuple[int, int]:
    opens = np.argwhere(grid == 0)
    if opens.size == 0:
        raise ValueError("Maze has no open cells.")
    idx = random.randrange(len(opens))
    r, c = opens[idx]
    return int(r), int(c)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolutionary maze training loop inspired by swarm-based learning.")
    parser.add_argument("--maze-dir", default=EvoConfig.maze_dir)
    parser.add_argument("--maze-id", default=EvoConfig.maze_id)
    parser.add_argument("--cycles", type=int, default=EvoConfig.cycles)
    parser.add_argument("--population", type=int, default=EvoConfig.population)
    parser.add_argument("--max-steps", type=int, default=EvoConfig.max_steps)
    parser.add_argument("--min-alive", type=int, default=EvoConfig.min_alive)
    parser.add_argument("--sensor-range", type=int, default=EvoConfig.sensor_range)
    parser.add_argument("--top-k", type=int, default=EvoConfig.top_k)
    parser.add_argument("--mutation-base", type=float, default=EvoConfig.mutation_base)
    parser.add_argument("--mutation-age", type=float, default=EvoConfig.mutation_age)
    parser.add_argument("--move-start-on-success", action="store_true", default=EvoConfig.move_start_on_success)
    parser.add_argument("--batch-size", type=int, default=EvoConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=EvoConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=EvoConfig.learning_rate)
    parser.add_argument("--model-path", default=EvoConfig.model_path)
    parser.add_argument("--log-dir", default=EvoConfig.log_dir)
    args = parser.parse_args()

    cfg = EvoConfig(
        maze_dir=args.maze_dir,
        maze_id=args.maze_id,
        cycles=args.cycles,
        population=args.population,
        max_steps=args.max_steps,
        min_alive=args.min_alive,
        sensor_range=args.sensor_range,
        top_k=args.top_k,
        mutation_base=args.mutation_base,
        mutation_age=args.mutation_age,
        move_start_on_success=args.move_start_on_success,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_path=args.model_path,
        log_dir=args.log_dir,
    )

    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"evo_run_{run_ts}.jsonl"
    metrics_path = log_dir / "evo_metrics.csv"

    maze_id = _resolve_maze_id(cfg.maze_dir, cfg.maze_id)
    grid, meta = load_maze_files(cfg.maze_dir, maze_id, prefer_npy=True)
    start = tuple(meta.get("start", _random_open_cell(grid)))
    goal = tuple(meta.get("goal", _random_open_cell(grid)))
    dist_map = _compute_distances(grid, goal)  # A* distance proxy

    model = EvoModel(input_size=8)
    model_path = Path(cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "config", "maze_id": maze_id, **cfg.__dict__}) + "\n")

    if not metrics_path.exists():
        metrics_path.write_text(
            "cycle,avg_steps,goal_rate,best_dist,avg_dist,train_rows,val_loss\n",
            encoding="utf-8",
        )

    for cycle in range(1, cfg.cycles + 1):
        agents = [Agent(start) for _ in range(cfg.population)]
        goal_reached = 0
        steps_sum = 0
        alive = cfg.population

        for _ in range(cfg.max_steps):
            if alive <= cfg.min_alive:
                break
            for agent in agents:
                if not agent.alive:
                    continue
                obs = _ray_sensors(grid, agent.pos, cfg.sensor_range)
                mutation_prob = cfg.mutation_base + (agent.steps / max(1, cfg.max_steps)) * cfg.mutation_age
                mutation_prob = max(0.0, min(1.0, mutation_prob))
                action = _choose_action(model, obs, mutation_prob, agent.prev_action)
                agent.trajectory.append((obs, action))
                moved = _step_agent(grid, agent, action)
                agent.steps += 1
                agent.prev_action = action
                if not moved:
                    alive -= 1
                    continue
                if agent.pos == goal:
                    goal_reached += 1
                    agent.alive = False
                    alive -= 1
            steps_sum += 1

        scored = []
        dist_vals = []
        for agent in agents:
            r, c = agent.pos
            dist = float(dist_map[r, c])
            if not np.isfinite(dist):
                dist = float("inf")
            scored.append((dist, agent))
            dist_vals.append(dist)
        scored.sort(key=lambda item: item[0])
        best_dist = scored[0][0] if scored else float("inf")
        avg_dist = float(np.mean([d for d in dist_vals if np.isfinite(d)])) if dist_vals else float("inf")

        top_agents = [agent for _, agent in scored[: cfg.top_k] if agent.trajectory]
        training_rows = []
        for agent in top_agents:
            training_rows.extend(agent.trajectory)
        val_loss = _train_model(
            model,
            training_rows,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
        )
        torch.save(model.state_dict(), model_path)

        goal_rate = goal_reached / max(1, cfg.population)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{cycle},{steps_sum:.0f},{goal_rate:.3f},{best_dist:.1f},{avg_dist:.1f},{len(training_rows)},{val_loss:.6f}\n"
            )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "cycle",
                        "cycle": cycle,
                        "avg_steps": steps_sum,
                        "goal_rate": goal_rate,
                        "best_dist": best_dist,
                        "avg_dist": avg_dist,
                        "train_rows": len(training_rows),
                        "val_loss": val_loss,
                    }
                )
                + "\n"
            )

        if goal_rate > 0 and cfg.move_start_on_success:
            start = _random_open_cell(grid)
            dist_map = _compute_distances(grid, goal)

    print(f"Saved model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
