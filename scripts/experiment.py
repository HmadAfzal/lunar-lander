import sys, csv, os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from wrappers import TimePenalty, HoverBonus, ProgressBonus

VARIANTS = {
    "baseline": (None, {}),
    "penalty": (TimePenalty, {"penalty": 0.05}),
    "hover": (HoverBonus, {}),
    "fixed": (ProgressBonus, {}),
}


def evaluate(model, variant, seed):
    env = Monitor(gym.make("LunarLander-v3"))  # wrapper OFF
    rewards, lengths = evaluate_policy(
        model, env, n_eval_episodes=20, return_episode_rewards=True
    )
    row = [variant, seed, np.mean(rewards), np.std(rewards), np.mean(lengths)]
    new = not os.path.exists("results/results.csv")
    with open("results/results.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["variant", "seed", "mean_reward", "std_reward", "mean_frames"])
        w.writerow(row)
    print(row)


def train(variant, seed):
    wrapper, kwargs = VARIANTS[variant]
    env = make_vec_env(
        "LunarLander-v3", n_envs=16, seed=seed,
        wrapper_class=wrapper, wrapper_kwargs=kwargs,
    )
    model = PPO(
        "MlpPolicy", env, seed=seed,
        n_steps=1024, batch_size=64, n_epochs=4,
        gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
    )
    model.learn(total_timesteps=1_000_000)
    model.save(f"models/ppo_{variant}_s{seed}")
    evaluate(model, variant, seed)


if __name__ == "__main__":
    mode, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if mode == "train":
        train(variant, seed)
    else:  # eval an existing model: python experiment.py eval hover 0 ppo_hover
        evaluate(PPO.load(sys.argv[4]), variant, seed)