import gymnasium as gym
import torch
from torch import optim
import torch.nn.functional as F

from typing import Dict

from jaxtyping import Float

from abc import ABC

from value_functions import (
    StateValueFunction,
)
from policies import DiscretePolicy, GaussianPolicy
from memory_classes import BatchStorage


import yaml


class OnPolicyAgent(ABC):
    def __init__(self, env: gym.Env, hyperparameters: Dict, defaults: Dict):

        self.gamma = hyperparameters.get("gamma", defaults["gamma"])
        self.batch_size = hyperparameters.get("batch_size", defaults["batch_size"])
        self.trajectories_per_epoch = hyperparameters.get(
            "trajectories_per_epoch", defaults["trajectories_per_epoch"]
        )
        self.normalise_advantages = hyperparameters.get(
            "normalise_advantages", defaults["normalise_advantages"]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_storage = BatchStorage(
            env,
            self.trajectories_per_epoch,
            self.batch_size,
            self.device,
        )

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.policy = DiscretePolicy(
                env,
                hyperparameters.get("actor_mlp_widths", defaults["actor_mlp_widths"]),
            ).to(self.device)
        elif isinstance(env.action_space, gym.spaces.Box):
            self.policy = GaussianPolicy(
                env,
                hyperparameters.get("actor_mlp_widths", defaults["actor_mlp_widths"]),
            ).to(self.device)

        self.value_function = StateValueFunction(
            env,
            hyperparameters.get("critic_mlp_widths", defaults["critic_mlp_widths"]),
        ).to(self.device)

        self.actor_optimiser = optim.Adam(self.policy.parameters(), maximize=True)
        self.critic_optimiser = optim.Adam(self.value_function.parameters())

    def empty_batch_storage(self):
        self.batch_storage.empty()

    def process_rollouts(self):
        """
        Process stored trajectories to compute returns and advantages.
        Uses backwards iteration to compute returns efficiently.
        """
        print("Processing rollouts")

        # Iterate through trajectories in reverse
        for trajectory in self.batch_storage.trajectories:

            # Get value estimates for all states in trajectory
            with torch.no_grad():
                states = trajectory.states.to(self.device)
                current_values = self.value_function(states)
                current_values = current_values.cpu()
                actions = trajectory.actions.to(self.device)
                action_probs = self.policy.get_probs(states, actions)
                action_probs = action_probs.cpu()
                next_states = trajectory.next_states.to(self.device)
                next_values = self.value_function(next_states)
                next_values = next_values.cpu()

            advantages = []
            returns = []
            one_step_returns = []

            if trajectory.dones[-1]:
                current_return = 0.0
            else:
                current_return = next_values[-1]

            for t in range(trajectory.length - 1, -1, -1):
                if trajectory.dones[t]:
                    next_state_value = 0.0
                else:
                    next_state_value = next_values[t]

                one_step_return = trajectory.rewards[t] + self.gamma * next_state_value

                # TODO: Compute the advantage estimate
                advantage = None

                # TODO: Compute the return G_t
                current_return = None

                advantages.append(advantage)
                returns.append(current_return)
                one_step_returns.append(one_step_return)

            # Reverse lists and convert to tensors
            one_step_returns = torch.tensor(one_step_returns[::-1], device=self.device)
            advantages = torch.tensor(advantages[::-1], device=self.device)
            returns = torch.tensor(returns[::-1], device=self.device)

            trajectory.advantages = advantages
            trajectory.action_probs = action_probs
            trajectory.regression_targets = returns
            print(f"{current_return=}")

    def update_value_function(self):
        """
        Update value function using L2 loss between predicted values
        and computed returns.
        """
        print("Training value function")
        self.batch_storage.start_async()

        # Train for specified number of epochs
        for _ in range(self.value_function_train_iterations):
            # Get minibatch
            batch = self.batch_storage.get_batch()

            # Compute value predictions
            values = self.value_function(batch["states"])

            # Get regression targets
            regression_targets = batch["regression_targets"]

            # TODO: Implement the value function loss
            value_loss = None

            # Optimization step
            self.critic_optimiser.zero_grad()
            value_loss.backward()
            self.critic_optimiser.step()

    def evaluation_policy(self, state: Float[torch.Tensor, "{self.state_dim}"]):
        with torch.no_grad():
            return self.policy.sample_action(state.to(self.device))

    def training_policy(self, state: Float[torch.Tensor, "{self.state_dim}"]):
        with torch.no_grad():
            return self.policy.sample_action(state.to(self.device))

    def generate_trajectories(self, env: gym.Env, num_trajectories: int):
        print("Generating on-policy trajectories")
        state, info = env.reset()
        for ii in range(num_trajectories):
            while True:
                action = self.training_policy(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                self.batch_storage.add(
                    state, action, reward, next_state, terminated, truncated
                )
                if terminated or truncated:
                    state, info = env.reset()
                    break
                else:
                    state = next_state

    def update_policy(self):
        pass

    def train(self, env: gym.Env, epochs: int):
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            self.generate_trajectories(env, self.trajectories_per_epoch)
            self.process_rollouts()
            self.update_policy()
            self.update_value_function()
            self.empty_batch_storage()


with open("a2c_defaults.yaml", "r") as file:
    a2c_defaults = yaml.safe_load(file)


class A2CAgent(OnPolicyAgent):
    def __init__(self, env: gym.Env, hyperparameters: Dict):
        self.additional_fields = ["advantage", "regression_target"]
        super().__init__(env, hyperparameters, a2c_defaults)

        self.value_function_train_iterations = hyperparameters.get(
            "value_function_train_iterations",
            a2c_defaults["value_function_train_iterations"],
        )

    def update_policy(self):
        """
        Update policy using A2C loss aggregated over all transitions.
        """
        print("Updating policy")
        states = []
        advantages = []
        actions = []

        # Aggregate all transitions
        for trajectory in self.batch_storage.trajectories:
            states.append(trajectory.states)
            advantages.append(trajectory.advantages)
            actions.append(trajectory.actions)

        # Concatenate all transitions
        states = torch.cat(states).to(self.device)
        advantages = torch.cat(advantages).to(self.device)
        actions = torch.cat(actions).to(self.device)

        # Normalize advantages
        if self.normalise_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # TODO: implement the A2C policy update

        # Get the action probabilities
        action_probs = None

        # Compute policy objective
        policy_objective = None

        # Optimization step
        self.actor_optimiser.zero_grad()
        policy_objective.backward()
        self.actor_optimiser.step()


with open("ppo_defaults.yaml", "r") as file:
    ppo_defaults = yaml.safe_load(file)


class PPOAgent(OnPolicyAgent):
    def __init__(self, env: gym.Env, hyperparameters: Dict):
        super().__init__(env, hyperparameters, ppo_defaults)

        self.clip_parameter = hyperparameters.get(
            "clip_parameter", ppo_defaults["clip_parameter"]
        )
        self.policy_train_iterations = hyperparameters.get(
            "policy_train_iterations", ppo_defaults["policy_train_iterations"]
        )

        self.value_function_train_iterations = hyperparameters.get(
            "value_function_train_iterations",
            ppo_defaults["value_function_train_iterations"],
        )

    def update_policy(self):
        print("Updating policy")
        self.batch_storage.start_async()

        # Train for specified number of epochs
        for _ in range(self.policy_train_iterations):

            batch = self.batch_storage.get_batch()

            old_action_probs = batch["action_probs"]
            assert old_action_probs.requires_grad() == False
            new_action_probs = self.policy.get_probs(batch["states"], batch["actions"])
            advantages = batch["advantages"]

            # TODO: Compute the ratio of new action probabilities to old action probabilities
            ratio = None

            # TODO: Compute the clipped and unclipped terms for the objective
            unclipped_term = None
            clipped_term = None

            policy_objective = torch.min(clipped_term, unclipped_term).mean()

            # Optimization step
            self.critic_optimiser.zero_grad()
            policy_objective.backward()
            self.critic_optimiser.step()
