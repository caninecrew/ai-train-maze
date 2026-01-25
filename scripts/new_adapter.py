import argparse
from pathlib import Path


TEMPLATE = """from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from games.base import GameAdapter


class {class_name}(gym.Env):
    metadata = {{"render_modes": ["rgb_array"]}}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self._obs(), {{}}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        return self._obs(), reward, terminated, truncated, {{}}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def _obs(self):
        return np.zeros((4,), dtype=np.float32)


def _make_env(render_mode: Optional[str], seed: Optional[int], variant: Optional[int]) -> gym.Env:
    env = {class_name}(render_mode=render_mode, seed=seed)
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
    return {{
        "avg_reward": avg_reward,
        "avg_reward_ci": 0.0,
        "avg_ep_len": avg_len,
        "avg_ep_len_ci": 0.0,
    }}


def adapter() -> GameAdapter:
    return GameAdapter(
        name="{adapter_name}",
        description="{description}",
        model_prefix="ppo_{adapter_name}",
        make_env_fn=_make_env,
        extra_metrics=[],
        eval_fn=_evaluate,
        heatmap_fn=None,
    )
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a new game adapter template.")
    parser.add_argument("name", help="Adapter name, e.g. breakout")
    parser.add_argument("--class-name", default=None, help="Python class name, e.g. BreakoutEnv")
    parser.add_argument("--description", default="Custom environment.", help="Short description")
    parser.add_argument("--out", default="games", help="Output folder")
    args = parser.parse_args()

    adapter_name = args.name.strip().lower().replace(" ", "_")
    class_name = args.class_name or "".join(part.capitalize() for part in adapter_name.split("_")) + "Env"
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{adapter_name}_adapter.py"

    if dest.exists():
        raise SystemExit(f"File already exists: {dest}")

    content = TEMPLATE.format(adapter_name=adapter_name, class_name=class_name, description=args.description)
    dest.write_text(content, encoding="utf-8")
    print(f"Wrote {dest}")
    print("Register it in games/registry.py and add a config under configs/.")


if __name__ == "__main__":
    main()
