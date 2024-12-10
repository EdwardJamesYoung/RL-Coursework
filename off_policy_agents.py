import gymnasium as gym
import torch
from torch import nn
from torch import optim

from typing import Dict

from jaxtyping import Integer, Float

from abc import ABC

from value_functions import (
    ContinuousActionValueFunction,
    DiscreteActionValueFunction,
)
from policies import DeterministicPolicy
from memory_classes import ReplayBuffer

from copy import deepcopy

import yaml


class OffPolicyAgent(ABC):
    def __init__(self, env: gym.Env, hyperparameters: Dict, defaults: Dict):
        self.gamma = hyperparameters.get("gamma", dqn_defaults["gamma"])
        self.batch_size = hyperparameters.get("batch_size", defaults["batch_size"])
        self.replay_buffer_size = hyperparameters.get(
            "replay_buffer_size", defaults["replay_buffer_size"]
        )
        self.learning_starts = hyperparameters.get(
            "learning_starts", defaults["learning_starts"]
        )
        self.update_frequency = hyperparameters.get(
            "update_frequency", defaults["update_frequency"]
        )
        self.target_update_frequency = hyperparameters.get(
            "target_update_frequency", defaults["target_update_frequency"]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(env, self.replay_buffer_size, self.batch_size)

    def target_update(self):
        pass

    def update(self):
        pass

    def train(self, env: gym.Env, total_steps: int):
        current_state, info = env.reset()
        step_idx = 0

        while step_idx < self.learning_starts:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            self.replay_buffer.add(
                current_state, action, reward, next_state, terminated
            )
            current_state = next_state
            step_idx += 1

        self.replay_buffer.start_async()

        for step_idx in range(total_steps):
            action = self.training_policy(current_state)
            next_state, reward, terminated, truncated, info = env.step(action)
            # Load the experience into the memory replay buffer
            self.replay_buffer.add(
                current_state, action, reward, next_state, terminated
            )

            if step_idx % self.update_frequency == 0:
                # TODO: update the network parameters
                pass

            if step_idx % self.target_update_frequency == 0:
                # TODO: update the target network parameters
                pass

            if terminated or truncated:
                current_state, info = env.reset()
            else:
                current_state = next_state


with open("dqn_defaults.yaml", "r") as file:
    dqn_defaults = yaml.safe_load(file)


class DQNAgent(OffPolicyAgent):
    def __init__(
        self,
        env: gym.Env,
        hyperparameters: Dict,
    ):
        super().__init__(env, hyperparameters, dqn_defaults)
        self.epsilon = hyperparameters.get("epsilon", dqn_defaults["epsilon"])
        self.lr = hyperparameters.get("lr", dqn_defaults["lr"])

        self.state_dim = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.action_value_network = DiscreteActionValueFunction(
            env, hyperparameters.get("mlp_widths", dqn_defaults["mlp_widths"])
        ).to(self.device)
        self.target_value_network = deepcopy(self.action_value_network)

        self.optimiser = optim.Adam(self.action_value_network.parameters(), lr=self.lr)

    def target_update(self):
        self.target_value_network.load_state_dict(
            self.action_value_network.state_dict()
        )

    def evaluation_policy(
        self, state: Float[torch.Tensor, "{self.state_dim}"]
    ) -> Integer[torch.Tensor, "1"]:
        # TODO: implement the greedy policy
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            # Compute the Q-values of the actions
            q_values = None
            # Compute the action with the largest Q-value
            argmax_action = None
            return argmax_action.item()

    def training_policy(
        self, state: Float[torch.Tensor, "{self.state_dim}"]
    ) -> Integer[torch.Tensor, "1"]:
        # Draw a random number between 0 and 1.
        rand_num = torch.rand(1).item()

        # TODO: Implement the epsilon greedy exploration policy
        if rand_num < self.epsilon:
            # Behave randomly
            pass
        else:
            # Behave greedily
            pass

    def compute_regression_targets(
        self, batch: Dict[str, torch.Tensor]
    ) -> Float[torch.Tensor, "batch 1"]:
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # TODO: implement the computation of regression targets
        with torch.no_grad():
            # Compute the Q-values of every action in the next state
            next_state_action_values = None
            # Compute the next state value V(s'), i.e., the maximal value of any action of the next state
            next_state_values = None
            # Compute the regression target, r + gamma (1 - d) V(s')
            regression_targets = None

        return regression_targets

    def update(self):
        batch = self.replay_buffer.get_batch()
        regression_targets = self.compute_regression_targets(batch)

        all_action_values = self.action_value_network(batch["states"])

        taken_action_values = all_action_values[
            range(self.batch_size), batch["actions"]
        ]

        # TODO: Implement the l2 loss function between the Q values and the regression targets
        loss = None
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


with open("ddpg_defaults.yaml", "r") as file:
    ddpg_defaults = yaml.safe_load(file)


class DDPGAgent(OffPolicyAgent):
    def __init__(
        self,
        env: gym.Env,
        hyperparameters: Dict,
    ):
        super().__init__(env, hyperparameters, ddpg_defaults)
        self.actor_lr = hyperparameters.get("actor_lr", ddpg_defaults["actor_lr"])
        self.critic_lr = hyperparameters.get("critic_lr", ddpg_defaults["critic_lr"])
        self.polyak_parameter = hyperparameters.get(
            "polyak_parameter", ddpg_defaults["polyak_parameter"]
        )
        self.action_noise_std = hyperparameters.get(
            "action_noise_std", ddpg_defaults["action_noise_std"]
        )

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.critic_mlp_widths = hyperparameters.get(
            "critic_mlp_widths", ddpg_defaults["critic_mlp_widths"]
        )

        self.action_value_network = ContinuousActionValueFunction(
            env,
            hyperparameters.get(
                "critic_mlp_widths", ddpg_defaults["critic_mlp_widths"]
            ),
        ).to(self.device)
        self.target_value_network = deepcopy(self.action_value_network)

        self.policy = DeterministicPolicy(
            env,
            hyperparameters.get("actor_mlp_widths", ddpg_defaults["actor_mlp_widths"]),
        ).to(self.device)
        self.target_policy = deepcopy(self.policy)

        self.target_update()

        self.actor_optimiser = optim.Adam(
            self.policy.parameters(), lr=self.actor_lr, maximize=True
        )
        self.critic_optimiser = optim.Adam(
            self.action_value_network.parameters(), lr=self.critic_lr
        )

    def target_update(self):
        for target_param, param in zip(
            self.target_value_network.parameters(),
            self.action_value_network.parameters(),
        ):
            # TODO: Implement Polyak averaging
            target_param.data = (
                self.polyak_parameter * None + (1 - self.polyak_parameter) * None
            )

    def evaluation_policy(
        self, state: Float[torch.Tensor, "{self.state_dim}"]
    ) -> Float[torch.Tensor, "{self.action_dim}"]:
        with torch.no_grad():
            return self.policy(state.to(self.device))

    def training_policy(
        self, state: Float[torch.Tensor, "{self.state_dim}"]
    ) -> Float[torch.Tensor, "{self.action_dim}"]:
        with torch.no_grad():
            noiseless_action = self.policy(state.to(self.device))
            noise = torch.normal(
                mean=0,
                std=self.action_noise_std,
                size=noiseless_action.shape,
                device=self.device,
            )
            # TODO: Implement the exploration policy
            noisy_action = None
            return noisy_action

    def compute_regression_targets(
        self, batch: Dict[str, torch.Tensor]
    ) -> Float[torch.Tensor, "batch"]:
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # TODO: Implement the regression target computation
        with torch.no_grad():
            # Get the actions taken at the next state, a' = mu(s')
            next_actions = None
            # Compute their values, V(s') = Q(s',a')
            next_state_values = None
            # Compute the regression target, r + gamma (1 - d) V(s')
            regression_targets = None

        return regression_targets

    def update(self):
        batch = self.replay_buffer.get_batch()

        regression_targets = self.compute_regression_targets(batch)
        taken_action_values = self.action_value_network(
            batch["states"], batch["actions"]
        )
        # TODO: Implement the loss function for the value network
        loss = None
        self.critic_optimiser.zero_grad()
        loss.backward()
        self.critic_optimiser.step()

        # TODO: Implement the policy objective for the policy network
        policy_objective = None

        self.actor_optimiser.zero_grad()
        policy_objective.backward()
        self.actor_optimiser.step()
