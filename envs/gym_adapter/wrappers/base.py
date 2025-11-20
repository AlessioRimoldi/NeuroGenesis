from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @abstractmethod
    def seed(self, seed: int): ...
    @abstractmethod
    def reset(self):
        """Returns observation"""
    @abstractmethod
    def step(self, action):
        """Returns (observation, reward, done, info)"""
    @abstractmethod
    def observation_space(self):...
    @abstractmethod
    def action_space(self): ...

