# Pong AI Trainer

Lightweight reinforcement learning setup for training PPO agents on a custom Pong environment built with pygame and Stable Baselines3.

## Requirements
- Python 3.9+
- `pip install -r requirements` equivalent packages: `pygame`, `gymnasium`, `stable-baselines3`, `imageio`, `Pillow`, `numpy`

## Usage
- Demo the environment: `python pong.py` (controls: W/S left, Up/Down right; auto-tracks when headless).
- Train agents: `python train_pong_ppo.py [--train-timesteps 200000 --checkpoint-interval 1 --seed 0 ...]`
  - Artifacts: models in `models/`, logs in `logs/`, combined videos in `videos/`.
  - Training run logs config and per-cycle metrics to `logs/train_run_<timestamp>.jsonl`.
- Evaluate a checkpoint: `python eval_pong.py --model-path models/ppo_pong_custom_latest.zip --episodes 5 [--render]`.

## Smoke Tests
- Quick checks for shapes, deterministic seeds, and paddle bounds: `python -m pytest tests/test_pong_env.py`.
