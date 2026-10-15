import gymnasium as gym
import numpy as np


class TimePenalty(gym.Wrapper):
    def __init__(self, env, penalty=0.05):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.penalty
        return obs, reward, terminated, truncated, info



class HoverBonus(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        angle = obs[4]
        vx, vy = obs[2], obs[3]
        if abs(angle) < 0.2 and abs(vx) < 0.3 and abs(vy) < 0.3:
            reward += 1.0
        return obs, reward, terminated, truncated, info

class ProgressBonus(gym.Wrapper):
    """Potential-based shaping + time penalty.
    Pays for the *improvement* in 'upright and slow' between frames,
    so total bonus over an episode telescopes to a small constant.
    Hovering earns nothing extra, and the time penalty makes it a loss."""

    def __init__(self, env, scale=1.0, penalty=0.05):
        super().__init__(env)
        self.scale = scale
        self.penalty = penalty
        self.prev_phi = 0.0

    @staticmethod
    def _phi(obs):
        # higher = more upright and slower
        return -(abs(obs[4]) + abs(obs[2]) + abs(obs[3]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = self._phi(obs)
        reward += self.scale * (phi - self.prev_phi) - self.penalty
        self.prev_phi = phi
        return obs, reward, terminated, truncated, info