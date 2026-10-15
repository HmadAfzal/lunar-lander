import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

name = sys.argv[1] if len(sys.argv) > 1 else "ppo_hover"
tag = name.replace("ppo_", "")

model = PPO.load(f"models/{name}")

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=f"videos/{tag}",
    episode_trigger=lambda ep: True,
    name_prefix=tag,
)

for ep in range(5):
    obs, info = env.reset(seed=ep)
    done = False
    total, frames = 0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        frames += 1
        done = terminated or truncated
    print(f"episode {ep}: reward {total:.1f}, frames {frames}")

env.close()