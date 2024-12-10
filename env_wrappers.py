import gymnasium as gym
import torch
import numpy as np
from typing import Any, Tuple, Dict


class PyTorchGymWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # Determine observation dtype based on environment
        if isinstance(env.observation_space, gym.spaces.Box):
            if np.issubdtype(env.observation_space.dtype, np.floating):
                self.obs_dtype = torch.float32
            else:
                self.obs_dtype = torch.uint8
        else:
            self.obs_dtype = torch.float32

    def reset(self, **kwargs) -> Tuple[torch.Tensor, Dict]:
        obs, info = self.env.reset(**kwargs)
        return torch.as_tensor(obs, dtype=self.obs_dtype), info

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, bool, Dict[str, Any]]:

        if isinstance(action, torch.Tensor):
            action = action.cpu()
            if self.is_discrete:
                action = action.item()
            else:
                action = action.detach().numpy()

        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert observations and rewards to tensors
        next_obs = torch.as_tensor(next_obs, dtype=self.obs_dtype)
        reward = torch.as_tensor([reward], dtype=torch.float32)

        return next_obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
