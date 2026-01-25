from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from games.base import GameAdapter


class CounterEnv(gym.Env):
    """
    Simple placeholder environment.
    - Observation: 4 floats (counter, action hint, noise, bias)
    - Action: Discrete(2)
    - Reward: +1 for matching the hint, else -1
    - Episode ends after max_steps
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None, max_steps: int = 50):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._hint = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._hint = int(self._rng.integers(0, 2))
        return self._obs(), {}

    def step(self, action):
        action_int = int(action)
        reward = 1.0 if action_int == self._hint else -1.0
        self._step += 1
        terminated = self._step >= self.max_steps
        truncated = False
        self._hint = int(self._rng.integers(0, 2))
        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        color = 200 if self._hint == 1 else 50
        frame[:, :, :] = color
        return frame

    def _obs(self):
        noise = float(self._rng.normal(0, 0.1))
        return np.array(
            [
                float(self._step) / float(self.max_steps),
                float(self._hint),
                noise,
                0.5,
            ],
            dtype=np.float32,
        )


def _make_env(render_mode: Optional[str], seed: Optional[int], variant: Optional[int]) -> gym.Env:
    max_steps = 50 if variant is None else 50 + int(variant)
    env = CounterEnv(render_mode=render_mode, seed=seed, max_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _evaluate(model: Any, episodes: int, deterministic: bool = True) -> Dict[str, float]:
    env = _make_env(None, None, None)
    rewards = []
    lengths = []
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
        "avg_reward_ci": 0.0,
        "avg_ep_len": avg_len,
        "avg_ep_len_ci": 0.0,
    }


def template_adapter() -> GameAdapter:
    return GameAdapter(
        name="template",
        description="Minimal placeholder environment to replace with your own.",
        model_prefix="ppo_template",
        make_env_fn=_make_env,
        extra_metrics=[],
        eval_fn=_evaluate,
        heatmap_fn=None,
    )
