import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F

from typing import Dict, Optional, List, Tuple

from jaxtyping import jaxtyped, Integer, Float
from typeguard import typechecked

from abc import ABC, abstractmethod

from helpers import get_mlp

import yaml

with open("ddpg_defaults.yaml", "r") as file:
    ddpg_defaults = yaml.safe_load(file)


class DeterministicPolicy(nn.Module):
    def __init__(self, env: gym.Env, mlp_widths: List[int]):
        super().__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.network = get_mlp(
            input_dim=self.state_dim,
            widths=mlp_widths,
            output_dim=self.action_dim,
            non_linearity="ReLU",
        )

    def forward(
        self, states: Float[torch.Tensor, "batch {self.state_dim}"]
    ) -> Float[torch.Tensor, "batch {self.action_dim}"]:
        return self.network(states)


class DiscretePolicy(nn.Module):
    def __init__(self, env: gym.Env, mlp_widths: List[int]):
        super().__init__()
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.network = get_mlp(
            input_dim=self.obs_dim,
            widths=mlp_widths,
            output_dim=self.n_actions,
            non_linearity="ReLU",
        )

    def forward(
        self, states: Float[torch.Tensor, "batch {self.obs_dim}"]
    ) -> Float[torch.Tensor, "batch {self.n_actions}"]:
        logits = self.network(states)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def sample_action(self, state: Float[torch.Tensor, "{self.obs_dim}"]) -> int:
        logits = self.network(state)
        probabilities = F.softmax(logits, dim=-1)
        action = torch.multinomial(probabilities, num_samples=1)
        return action.cpu()

    def get_probs(
        self,
        states: Float[torch.Tensor, "batch {self.obs_dim}"],
        actions: Integer[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch"]:
        logits = self.network(states)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities.gather(dim=1, index=actions).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, env: gym.Env, mlp_widths: List[int]):
        super().__init__()
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.network = get_mlp(
            input_dim=self.obs_dim,
            widths=mlp_widths,
            output_dim=None,
            non_linearity="ReLU",
        )

        self.mean_head = nn.Linear(mlp_widths[-1], self.action_dim)
        self.log_std_head = nn.Linear(mlp_widths[-1], self.action_dim)

    def forward(self, states: Float[torch.Tensor, "batch {self.obs_dim}"]) -> Tuple[
        Float[torch.Tensor, "batch {self.action_dim}"],
        Float[torch.Tensor, "batch {self.action_dim}"],
    ]:
        x = self.network(states)
        means = self.mean_head(x)
        log_stds = self.log_std_head(x)
        stds = torch.exp(log_stds)
        return means, stds

    def sample_action(
        self, state: Float[torch.Tensor, "{self.obs_dim}"]
    ) -> Float[torch.Tensor, "{self.action_dim}"]:
        means, stds = self.forward(state)
        normal = torch.distributions.Normal(means, stds)
        action = normal.sample().cpu()
        return action

    def get_probs(
        self,
        states: Float[torch.Tensor, "batch {self.obs_dim}"],
        actions: Float[torch.Tensor, "batch {self.action_dim}"],
    ) -> Float[torch.Tensor, "batch"]:
        means, stds = self.forward(states)
        normal = torch.distributions.Normal(means, stds)
        log_probs = normal.log_prob(actions)
        per_action_log_probs = log_probs.sum(axis=-1)
        per_action_probs = torch.exp(per_action_log_probs)
        return per_action_probs
