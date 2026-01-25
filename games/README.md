# Game Adapters

Adapters are the contract between your environment and the training loop.

Checklist
- Implement `make_env(render_mode=None, seed=None, variant=None)` and return a `gymnasium.Env`.
- Ensure `reset()` returns `(obs, info)` and `step()` returns `(obs, reward, terminated, truncated, info)`.
- Define stable `observation_space` and `action_space`.
- Support `render_mode="rgb_array"` if you want videos; otherwise return `None`.
- Optional: implement custom metrics in `eval_fn` and add them to `extra_metrics`.
- Optional: implement `heatmap_fn` for the dashboard.

Minimal flow
1) Add a new adapter file in `games/`.
2) Register it in `games/registry.py`.
3) Add a config under `configs/`.
4) Run `python train.py --game your_game --config configs/your_game.yaml`.
