# Quick Guide

This repo is a base for training an AI on any environment that can be exposed as a Gymnasium-style `Env`.

## Bring your own environment
You can wrap an existing Gymnasium env or build a custom one. The key is to implement:
- `reset()` -> `(obs, info)`
- `step(action)` -> `(obs, reward, terminated, truncated, info)`
- `observation_space` and `action_space`

## Example: wrap an existing Gymnasium env
```python
import gymnasium as gym
from games.base import GameAdapter

def _make_env(render_mode=None, seed=None, variant=None):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env

def adapter():
    return GameAdapter(
        name="cartpole",
        description="Classic control task.",
        model_prefix="ppo_cartpole",
        make_env_fn=_make_env,
    )
```

## Example: custom environment
Use `games/template_adapter.py` as the starting point. Update the env and evaluation logic, then register it.

## Register the adapter
Add your adapter to `games/registry.py`:
```python
from games.cartpole_adapter import cartpole_adapter

_REGISTRY = {
    "cartpole": cartpole_adapter(),
}
```

## Train and evaluate
```
python train.py --game cartpole --config configs/cartpole.yaml
python eval.py --game cartpole --model-path models/ppo_cartpole_latest.zip --episodes 5
```
