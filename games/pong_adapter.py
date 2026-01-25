from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from pong import PongEnv, simple_tracking_policy, Action, STAY
from games.base import GameAdapter


class SB3PongEnv(gym.Env):
    """
    Gymnasium wrapper around the custom PongEnv.
    The learning agent controls the left paddle; the right paddle uses a fixed policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        opponent_policy: Optional[Callable[[tuple, bool], Action]] = None,
        render_mode: Optional[str] = None,
        ball_color: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.env = PongEnv(render_mode=render_mode, ball_color=ball_color)
        self.opponent_policy = opponent_policy or (lambda obs, is_left: STAY)
        self.last_obs: Optional[tuple] = None

        # Observations are normalized: [bx, by, bvx, bvy, ly, ry]
        low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.env.reset()
        self.last_obs = obs
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        if self.last_obs is None:
            raise RuntimeError("Call reset() before step().")

        action_int = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        right_action = self.opponent_policy(self.last_obs, False)
        obs, reward, done, info = self.env.step(action_int, right_action)
        self.last_obs = obs

        terminated = False  # Episodes end only by truncation (step cap) in PongEnv.
        truncated = bool(done)
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def _make_env(render_mode: Optional[str], seed: Optional[int], variant: Optional[int]) -> gym.Env:
    ball_colors = [
        (255, 0, 0),
        (0, 200, 255),
        (255, 200, 0),
        (0, 255, 120),
    ]
    color = ball_colors[variant % len(ball_colors)] if variant is not None else None
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=render_mode, ball_color=color)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _evaluate(model: PPO, episodes: int, deterministic: bool = True) -> Dict[str, float]:
    eval_env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
    rewards = []
    returns = []
    rally_lengths = []
    lengths = []
    wins = 0
    for _ in range(episodes):
        obs, info = eval_env.reset()
        left_score = info.get("left_score", 0)
        right_score = info.get("right_score", 0)
        done = False
        ep_rew = 0.0
        steps = 0
        last_left = 0
        last_right = 0
        rally_steps = 0
        while not done and steps < eval_env.env.cfg.max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, terminated, truncated, info = eval_env.step(action)
            rally_steps += 1
            left_score = info.get("left_score", left_score)
            right_score = info.get("right_score", right_score)
            if left_score != last_left or right_score != last_right:
                rally_lengths.append(rally_steps)
                rally_steps = 0
                last_left, last_right = left_score, right_score
            if rew > 0:
                returns.append(1)
            ep_rew += rew
            steps += 1
            done = terminated or truncated
        rewards.append(ep_rew)
        lengths.append(steps)
        if left_score > right_score:
            wins += 1
    eval_env.close()

    def _ci(arr):
        if len(arr) < 2:
            return 0.0
        std = float(np.std(arr, ddof=1))
        return 1.96 * std / np.sqrt(len(arr))

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_return_rate = float(np.mean(returns)) if returns else 0.0
    avg_rally = float(np.mean(rally_lengths)) if rally_lengths else 0.0
    return {
        "avg_reward": avg_reward,
        "avg_reward_ci": _ci(rewards),
        "avg_ep_len": float(np.mean(lengths)) if lengths else 0.0,
        "avg_ep_len_ci": _ci(lengths),
        "avg_return_rate": avg_return_rate,
        "avg_return_rate_ci": _ci(returns),
        "avg_rally_length": avg_rally,
        "win_rate": wins / episodes if episodes else 0.0,
    }


def _heatmap(model_path: str, steps: int = 1500, bins: int = 40) -> Dict[str, Any]:
    from stable_baselines3 import PPO

    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
    try:
        model = PPO.load(model_path, env=env, device="cpu")
        obs, _ = env.reset()
        heat_ball = np.zeros((bins, bins), dtype=np.int32)
        heat_left = np.zeros((bins, bins), dtype=np.int32)
        heat_right = np.zeros((bins, bins), dtype=np.int32)
        heat_hits = np.zeros((bins, bins), dtype=np.int32)
        heat_scores = np.zeros((bins, bins), dtype=np.int32)
        last_left = 0
        last_right = 0
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            bx, by = obs[0], obs[1]
            ly, ry = obs[4], obs[5]
            x = min(bins - 1, max(0, int(bx * bins)))
            y = min(bins - 1, max(0, int(by * bins)))
            heat_ball[y, x] += 1
            ly_idx = min(bins - 1, max(0, int(ly * bins)))
            ry_idx = min(bins - 1, max(0, int(ry * bins)))
            heat_left[ly_idx, 1] += 1
            heat_right[ry_idx, bins - 2] += 1
            if reward > 0 and reward < 0.5:
                heat_hits[y, x] += 1
            left_score = info.get("left_score", last_left)
            right_score = info.get("right_score", last_right)
            if left_score != last_left or right_score != last_right:
                heat_scores[y, x] += 1
                last_left, last_right = left_score, right_score
            if terminated or truncated:
                obs, _ = env.reset()
        return {
            "ball": heat_ball.tolist(),
            "paddles": (heat_left + heat_right).tolist(),
            "hits": heat_hits.tolist(),
            "scores": heat_scores.tolist(),
        }
    finally:
        env.close()


def pong_adapter() -> GameAdapter:
    return GameAdapter(
        name="pong",
        description="Custom Pong environment (pygame).",
        model_prefix="ppo_pong_custom",
        make_env_fn=_make_env,
        extra_metrics=["win_rate", "avg_return_rate", "avg_return_rate_ci", "avg_rally_length"],
        eval_fn=_evaluate,
        heatmap_fn=_heatmap,
    )
