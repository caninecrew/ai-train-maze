import argparse
import os
import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pong import PongEnv, simple_tracking_policy, STAY
from train_pong_ppo import SB3PongEnv, evaluate_model


def evaluate(model_path: str, episodes: int, render: bool) -> None:
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="human" if render else None)
    model = PPO.load(model_path, env=env)
    metrics = evaluate_model(model, episodes, deterministic=True)
    env.close()

    print(f"\nResults over {episodes} episodes")
    print(f"- Average reward: {metrics['avg_reward']:.3f} +/- {metrics['avg_reward_ci']:.3f}")
    print(f"- Win rate: {metrics['win_rate']:.2f}")
    print(f"- Average rally length: {metrics['avg_rally_length']:.2f}")
    print(f"- Average ball returns: {metrics['avg_return_rate']:.2f} +/- {metrics['avg_return_rate_ci']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Pong PPO checkpoint.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_pong_custom_latest.zip",
        help="Path to the PPO checkpoint to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render a human-visible window during evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV to append metrics to.")
    parser.add_argument("--deterministic", action="store_true", help="Toggle deterministic torch ops where possible.")
    parser.add_argument("--compare", nargs="*", default=None, help="Compare multiple checkpoints head-to-head.")
    parser.add_argument("--plot-path", type=str, default=None, help="Optional bar plot output when comparing.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        import torch

        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)

    if not Path(args.model_path).exists():
        print(f"Model not found at {args.model_path}. Exiting cleanly.")
        return

    def _run_single(path: str):
        try:
            env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="human" if args.render else None)
            model = PPO.load(path, env=env, device=args.device)
            metrics = evaluate_model(model, args.episodes, deterministic=args.deterministic)
            env.close()
            print(f"\nResults over {args.episodes} episodes for {path}")
            print(f"- Average reward: {metrics['avg_reward']:.3f} +/- {metrics['avg_reward_ci']:.3f}")
            print(f"- Win rate: {metrics['win_rate']:.2f}")
            print(f"- Average rally length: {metrics['avg_rally_length']:.2f}")
            print(f"- Average ball returns: {metrics['avg_return_rate']:.2f} +/- {metrics['avg_return_rate_ci']:.3f}")
        except FileNotFoundError:
            print(f"Could not load model from {path}")
            return

    if args.compare:
        model_paths = args.compare
        results = []
        for path in model_paths:
            if not Path(path).exists():
                print(f"Skipping missing model {path}")
                continue
            env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
            model = PPO.load(path, env=env, device=args.device)
            metrics = evaluate_model(model, args.episodes, deterministic=args.deterministic)
            results.append((path, metrics))
            env.close()
        print("\n=== Comparison ===")
        for path, metrics in results:
            print(f"{Path(path).name}: avg_reward={metrics['avg_reward']:.3f} win={metrics['win_rate']:.2f}")
        if len(results) >= 2:
            print("\nHead-to-head (left vs right):")
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    left_path, _ = results[i]
                    right_path, _ = results[j]
                    left_model = PPO.load(left_path, device=args.device)
                    right_model = PPO.load(right_path, device=args.device)
                    def _opp(obs, is_left, m=right_model):
                        act, _ = m.predict(obs, deterministic=args.deterministic)
                        return int(act)
                    env = SB3PongEnv(opponent_policy=_opp, render_mode=None)
                    metrics = evaluate_model(PPO.load(left_path, env=env, device=args.device), args.episodes, deterministic=args.deterministic)
                    env.close()
                    print(f"{Path(left_path).name} vs {Path(right_path).name}: win={metrics['win_rate']:.2f} reward={metrics['avg_reward']:.3f}")
        if args.plot_path and results:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                labels = [Path(p).name for p, _ in results]
                rewards = [m["avg_reward"] for _, m in results]
                plt.bar(labels, rewards)
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("Average Reward")
                plt.tight_layout()
                plt.savefig(args.plot_path)
                print(f"Saved comparison plot to {args.plot_path}")
            except Exception as exc:  # pragma: no cover - optional dep
                print(f"Could not write plot: {exc}")
    else:
        _run_single(args.model_path)

    if args.output_csv:
        metrics = evaluate_model(PPO.load(args.model_path, device=args.device), args.episodes, deterministic=args.deterministic)
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        headers = ["model_path", "episodes", "avg_reward", "avg_reward_ci", "win_rate", "avg_return_rate", "avg_return_rate_ci", "avg_rally_length"]
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "model_path": args.model_path,
                    "episodes": args.episodes,
                    **metrics,
                }
            )


if __name__ == "__main__":
    main()
