# Lunar Lander: Reward Hacking and Reward Shaping

Training a PPO agent to land a rocket in Gymnasium's `LunarLander-v3`, then writing my own reward wrapper, breaking it on purpose, and fixing it.

The short version: a "+1 per frame while upright and slow" bonus makes the agent hover until the time limit instead of landing. Paying for *progress* instead of *position* fixes it.

## Results

All scores are measured on the **plain environment with my reward wrapper off**, 20 episodes per run, so my own bonus can't inflate the numbers. Training: PPO (Stable-Baselines3, 2x64 MLP), 1M timesteps, 16 parallel envs.

| variant        | seeds | true-env reward | frames / episode |
|:---------------|------:|----------------:|-----------------:|
| baseline       |     1 |           250.2 |              357 |
| penalty        |     1 |           263.1 |              308 |
| hover (broken) |     3 |     -24.1 ± 74.5 |              999 |
| fixed          |     3 |      263.0 ± 13.8 |              306 |

Landing counts as solved at 200+.

**Read the frames column first.** The hover agent's reward swings between about -109 and +31 across seeds, so reward alone is noisy. Frames is not: every seed sits at the 1000-frame time limit. It never lands, it just floats.

Baseline and penalty are single-seed runs. Hover and fixed have 3 seeds each, since that contrast is the point of the project.

## Clips

- `videos/baseline/`: the default reward lands in about 6 seconds
- `videos/hover/`: the broken reward, hovering for the full 20 seconds
- `videos/fixed/`: the fixed reward lands cleanly again

## The experiment

Lunar Lander's built-in reward is already potential-based: each frame pays the *improvement* in a "how good is this position" score, minus fuel, then +/-100 for landing or crashing.

1. **Baseline**: built-in reward only.
2. **Penalty** (`TimePenalty`): subtract 0.05 per frame so waiting costs something. Still lands.
3. **Hover bonus** (`HoverBonus`), the deliberately bad reward: +1 per frame while `|angle| < 0.2`, `|vx| < 0.3`, `|vy| < 0.3`. An episode is cut off at 1000 frames, so hovering can collect up to about 1000 points, while landing ends the episode and stops the payments. PPO learns exactly that.
4. **Fixed** (`ProgressBonus`): pay for the *change* in an "upright and slow" score between frames, plus the 0.05 per-frame penalty.

```python
phi = -(abs(angle) + abs(vx) + abs(vy))    # higher = more upright and slow
reward += scale * (phi - prev_phi) - penalty
```

Over a whole episode the bonus telescopes to `phi_end - phi_start`, so sitting in a good pose earns roughly nothing and every extra frame costs the penalty. The only way to come out ahead is to improve and then finish.

**Lesson:** reward change, not state. A reward for sitting in a good spot teaches the agent to sit there forever.

## Setup

Needs Python 3.12 (box2d and PyTorch don't build cleanly on newer versions), `swig`, and a C++ compiler.

```bash
# Arch / Manjaro
sudo pacman -S swig gcc

uv python pin 3.12
uv sync
```

## Usage

Run everything from the project root.

```bash
# watch a random agent crash
uv run python scripts/random_agent.py

# train a variant: baseline | penalty | hover | fixed, then a seed
uv run python scripts/experiment.py train fixed 0

# evaluate an existing model and log it to results/results.csv
uv run python scripts/experiment.py eval hover 0 models/ppo_hover

# record 5 episodes to videos/<name>/
uv run python scripts/record.py ppo_fixed

# print the results table
uv run python scripts/table.py
```

Each training run takes roughly 25 minutes on CPU. No GPU needed: the network is tiny and the bottleneck is the Box2D physics.

## Layout

```
scripts/     wrappers, training/eval, recording, results table
models/      saved PPO models (gitignored)
results/     results.csv
videos/      baseline, hover, fixed clips
```

## Notes

- PPO hyperparameters (`n_steps=1024`, `batch_size=64`, `n_epochs=4`, `gamma=0.999`, `gae_lambda=0.98`, `ent_coef=0.01`) are the known-good set for this environment.
- Older tutorials use `LunarLander-v2`. Current Gymnasium only has `v3`.