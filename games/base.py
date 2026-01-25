from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np


def _confidence_interval(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    return 1.96 * std / np.sqrt(len(values))


def evaluate_generic(
    model: Any,
    make_env: Callable[[Optional[str], Optional[int], Optional[int]], gym.Env],
    episodes: int,
    deterministic: bool = True,
) -> Dict[str, float]:
    env = make_env(None, None, None)
    rewards: List[float] = []
    lengths: List[int] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(steps)
    env.close()

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    return {
        "avg_reward": avg_reward,
        "avg_reward_ci": _confidence_interval(rewards),
        "avg_ep_len": avg_len,
        "avg_ep_len_ci": _confidence_interval(lengths),
    }


@dataclass
class GameAdapter:
    name: str
    description: str
    model_prefix: str
    make_env_fn: Callable[[Optional[str], Optional[int], Optional[int]], gym.Env]
    extra_metrics: List[str] = field(default_factory=list)
    eval_fn: Optional[Callable[[Any, int, bool], Dict[str, float]]] = None
    heatmap_fn: Optional[Callable[[str, int, int], Dict[str, Any]]] = None

    def make_env(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        variant: Optional[int] = None,
    ) -> gym.Env:
        return self.make_env_fn(render_mode, seed, variant)

    def evaluate(self, model: Any, episodes: int, deterministic: bool = True) -> Dict[str, float]:
        if self.eval_fn:
            return self.eval_fn(model, episodes, deterministic)
        return evaluate_generic(model, self.make_env, episodes, deterministic)

    def heatmap_from_model(self, model_path: str, steps: int = 1500, bins: int = 40) -> Dict[str, Any]:
        if not self.heatmap_fn:
            return {}
        return self.heatmap_fn(model_path, steps, bins)
