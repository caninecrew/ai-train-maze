import argparse
import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from games.registry import get_game, list_games


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument("--game", type=str, default="pong", help="Game key to evaluate.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the PPO checkpoint to evaluate.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render a human-visible window during evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV to append metrics to.")
    parser.add_argument("--deterministic", action="store_true", help="Toggle deterministic torch ops where possible.")
    parser.add_argument("--compare", nargs="*", default=None, help="Compare multiple checkpoints.")
    parser.add_argument("--plot-path", type=str, default=None, help="Optional bar plot output when comparing.")
    parser.add_argument("--list-games", action="store_true", help="List available games and exit.")
    args = parser.parse_args()

    if args.list_games:
        print("Available games:")
        for game in list_games():
            print(f"- {game.name}: {game.description}")
        return

    game = get_game(args.game)
    default_path = Path("models") / f"{game.model_prefix}_latest.zip"
    model_path = Path(args.model_path) if args.model_path else default_path

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        import torch

        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)

    if not model_path.exists():
        print(f"Model not found at {model_path}. Exiting cleanly.")
        return

    def _report_metrics(path: Path, metrics: dict, label: str) -> None:
        print(f"\nResults over {args.episodes} episodes for {label}")
        print(f"- Average reward: {metrics.get('avg_reward', 0.0):.3f} +/- {metrics.get('avg_reward_ci', 0.0):.3f}")
        if "win_rate" in metrics:
            print(f"- Win rate: {metrics.get('win_rate', 0.0):.2f}")
        if "avg_ep_len" in metrics:
            print(f"- Average episode length: {metrics.get('avg_ep_len', 0.0):.2f}")
        if "avg_return_rate" in metrics:
            print(f"- Average return rate: {metrics.get('avg_return_rate', 0.0):.2f} +/- {metrics.get('avg_return_rate_ci', 0.0):.3f}")
        if "avg_rally_length" in metrics:
            print(f"- Average rally length: {metrics.get('avg_rally_length', 0.0):.2f}")

    def _run_single(path: Path) -> None:
        env = game.make_env(render_mode="human" if args.render else None)
        model = PPO.load(str(path), env=env, device=args.device)
        metrics = game.evaluate(model, args.episodes, deterministic=args.deterministic)
        env.close()
        _report_metrics(path, metrics, str(path))

    if args.compare:
        model_paths = [Path(p) for p in args.compare]
        results = []
        for path in model_paths:
            if not path.exists():
                print(f"Skipping missing model {path}")
                continue
            env = game.make_env(render_mode=None)
            model = PPO.load(str(path), env=env, device=args.device)
            metrics = game.evaluate(model, args.episodes, deterministic=args.deterministic)
            results.append((path, metrics))
            env.close()
        print("\n=== Comparison ===")
        for path, metrics in results:
            print(f"{path.name}: avg_reward={metrics.get('avg_reward', 0.0):.3f}")
        if args.plot_path and results:
            try:
                import matplotlib.pyplot as plt  # type: ignore

                labels = [p.name for p, _ in results]
                rewards = [m.get("avg_reward", 0.0) for _, m in results]
                plt.bar(labels, rewards)
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("Average Reward")
                plt.tight_layout()
                plt.savefig(args.plot_path)
                print(f"Saved comparison plot to {args.plot_path}")
            except Exception as exc:  # pragma: no cover - optional dep
                print(f"Could not write plot: {exc}")
    else:
        _run_single(model_path)

    if args.output_csv:
        metrics = game.evaluate(PPO.load(str(model_path), device=args.device), args.episodes, deterministic=args.deterministic)
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        headers = ["model_path", "episodes", "avg_reward", "avg_reward_ci", "avg_ep_len", "avg_ep_len_ci"] + list(game.extra_metrics)
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            if not exists:
                writer.writeheader()
            row = {"model_path": str(model_path), "episodes": args.episodes}
            for key in headers:
                if key in ("model_path", "episodes"):
                    continue
                row[key] = metrics.get(key, "")
            writer.writerow(row)


if __name__ == "__main__":
    main()
