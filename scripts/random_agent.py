import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=0)

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 20 == 0:
        print(f"step {step:3d} | obs={obs.round(2)} | reward={reward:.2f}")

    if terminated or truncated:
        print("episode over, resetting")
        obs, info = env.reset()

env.close()