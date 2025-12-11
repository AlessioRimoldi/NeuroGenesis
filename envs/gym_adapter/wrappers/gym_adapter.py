from typing import Optional
import gymnasium as gym

from .base import BaseAdapter


class GymAdapter(BaseAdapter):
    def __init__(self, env_id: str, render_mode: Optional[str] = None):
        self._env = gym.make(env_id, render_mode=render_mode)

    def seed(self, seed: int):
        self._env.reset(seed=seed)

    def reset(self):
        obs, info = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def observation_space(self):
        return self._env.observation_space

    def action_space(self):
        return self._env.action_space
