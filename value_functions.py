import gymnasium as gym
import torch
from torch import nn

from typing import Dict, Optional, List

from jaxtyping import jaxtyped, Integer, Float
from typeguard import typechecked

from helpers import get_mlp


class StateValueFunction(nn.Module):
    def __init__(self, env: gym.Env, mlp_widths: List[int]):
        super().__init__()
        self.state_dim = env.observation_space.shape[0]

        self.network = get_mlp(
            input_dim=self.state_dim,
            widths=mlp_widths,
            output_dim=1,
            non_linearity="ReLU",
        )

    def forward(
        self, states: Float[torch.Tensor, "batch {self.state_dim}"]
    ) -> Float[torch.Tensor, "batch"]:
        value_predictions = self.network(states).squeeze(-1)
        return value_predictions


class ContinuousActionValueFunction(nn.Module):
    def __init__(self, env: gym.Env, mlp_widths: List[int]):
        super().__init__()

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.network = get_mlp(
            input_dim=self.state_dim + self.action_dim,
            widths=mlp_widths,
            output_dim=1,
            non_linearity="ReLU",
        )

    def forward(
        self,
        states: Float[torch.Tensor, "batch {self.state_dim}"],
        actions: Float[torch.Tensor, "batch {self.action_dim}"],
    ) -> Float[torch.Tensor, "batch"]:
        network_input = torch.cat([states, actions], dim=-1)
        value_predictions = self.network(network_input)
        value_predictions = value_predictions.squeeze(-1)
        return value_predictions


class DiscreteActionValueFunction(nn.Module):
    def __init__(self, env: gym.Env, mlp_widths: List[int]):
        super().__init__()

        self.state_dim = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.network = get_mlp(
            input_dim=self.state_dim,
            widths=mlp_widths,
            output_dim=self.num_actions,
            non_linearity="ReLU",
        )

    def forward(
        self, states: Float[torch.Tensor, "batch {self.state_dim}"]
    ) -> Float[torch.Tensor, "batch {self.num_actions}"]:
        action_values = self.network(states)
        return action_values
