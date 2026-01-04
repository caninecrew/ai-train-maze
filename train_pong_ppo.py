import os
import ale_py  # registers ALE with Gymnasium
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_atari_env("ALE/Pong-v5", n_envs=8, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="logs",
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        learning_rate=2.5e-4,
    )

    model.learn(total_timesteps=1_000_000)
    model.save("models/ppo_pong")
    env.close()

    print("Saved: models/ppo_pong.zip")

if __name__ == "__main__":
    main()