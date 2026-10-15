import pandas as pd

df = pd.read_csv("results/results.csv")
t = df.groupby("variant").agg(
    seeds=("seed", "count"),
    reward=("mean_reward", "mean"),
    reward_sd=("mean_reward", "std"),
    frames=("mean_frames", "mean"),
).round(1)

t = t.reindex(["baseline", "penalty", "hover", "fixed"])
print(t.to_markdown())