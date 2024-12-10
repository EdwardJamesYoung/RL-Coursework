import gymnasium as gym
import torch
import numpy as np

from typing import Dict, Optional, Union, List, Tuple, Iterator

from jaxtyping import jaxtyped, Integer, Float
from collections import deque
from dataclasses import dataclass
from typeguard import typechecked


class ReplayBuffer:
    def __init__(
        self,
        env: gym.Env,
        capacity: int,
        batch_size: int,
        training_device: Optional[torch.device] = None,
    ):
        self.capacity = capacity
        self.device = training_device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.state_dim = env.observation_space.shape[0]

        # Detect action space type and set appropriate dtype
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.action_dim = 1
            self.action_dtype = torch.long
        else:
            self.action_dim = env.action_space.shape[0]
            self.action_dtype = torch.float32

        self.batch_size = batch_size
        self.states = torch.zeros((capacity, self.state_dim), pin_memory=True)
        self.next_states = torch.zeros((capacity, self.state_dim), pin_memory=True)
        if self.is_discrete:
            self.actions = torch.zeros(capacity, dtype=torch.long, pin_memory=True)
        else:
            self.actions = torch.zeros(
                (capacity, self.action_dim), dtype=torch.float32, pin_memory=True
            )
        self.rewards = torch.zeros(capacity, pin_memory=True)
        self.dones = torch.zeros(capacity, pin_memory=True)
        self.idx = 0
        self.size = 0

        if torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.transfer_stream = None
        self.prefetch_batch: Optional[Dict[str, torch.Tensor]] = None

    @jaxtyped(typechecker=typechecked)
    def add(
        self,
        observation: Float[torch.Tensor, "{self.state_dim}"],
        action: Union[int, Float[torch.Tensor, "{self.action_dim}"]],
        reward: Float[torch.Tensor, "1"],
        next_observation: Float[torch.Tensor, "{self.state_dim}"],
        done: bool,
    ):
        self.states[self.idx] = observation
        self.next_states[self.idx] = next_observation

        if self.is_discrete:
            if isinstance(action, torch.Tensor):
                self.actions[self.idx] = action.long()
            else:
                self.actions[self.idx] = torch.tensor(action, dtype=torch.long)
        else:
            if isinstance(action, torch.Tensor):
                self.actions[self.idx] = action.float()
            else:
                self.actions[self.idx] = torch.tensor(action, dtype=torch.float32)

        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_async(self):
        if self.transfer_stream is None:
            self.prefetch_batch = self._sample_sync()
            return

        with torch.cuda.stream(self.transfer_stream):

            indices = torch.randint(0, self.size, (self.batch_size,))

            # Transfer and normalize states
            states = self.states[indices].to(self.device, non_blocking=True)
            next_states = self.next_states[indices].to(self.device, non_blocking=True)

            # Transfer other data, ensuring actions maintain correct dtype
            actions = self.actions[indices].to(
                self.device, dtype=self.action_dtype, non_blocking=True
            )
            rewards = self.rewards[indices].float().to(self.device, non_blocking=True)
            dones = self.dones[indices].float().to(self.device, non_blocking=True)

            self.prefetch_batch = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "dones": dones,
            }

    def _sample_sync(self) -> Dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (self.batch_size,))
        return {
            "states": self.states[indices].to(self.device),
            "actions": self.actions[indices].to(self.device, dtype=self.action_dtype),
            "rewards": self.rewards[indices].to(self.device),
            "next_states": self.next_states[indices].to(self.device),
            "dones": self.dones[indices].to(self.device),
        }

    @typechecked
    def get_batch(self) -> Dict[str, torch.Tensor]:
        if self.transfer_stream is not None:
            torch.cuda.current_stream().wait_stream(self.transfer_stream)

        batch = self.prefetch_batch
        self.sample_async()
        return batch

    def __len__(self) -> int:
        return self.size

    def start_async(self) -> None:
        self.sample_async()


@dataclass
class Trajectory:
    """Represents a single trajectory of experiences"""

    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    length: int
    regression_targets: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    action_probs: Optional[torch.Tensor]


class BatchStorage:
    def __init__(
        self,
        env: gym.Env,
        num_trajectories: int,
        batch_size: int,
        training_device: Optional[torch.device] = None,
    ):
        """
        Initialize BatchStorage for on-policy methods.

        Args:
            env: The gymnasium environment
            num_trajectories: Maximum number of trajectories to store
            batch_size: Size of minibatches for training
            additional_fields: List of additional fields to store (e.g., ["regression_target", "advantage"])
            training_device: Device to store tensors on
        """
        self.device = training_device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.state_dim = env.observation_space.shape[0]
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        if self.is_discrete:
            self.action_dim = 1
            self.action_dtype = torch.long
        else:
            self.action_dim = env.action_space.shape[0]
            self.action_dtype = torch.float32

        self.num_trajectories = num_trajectories
        self.batch_size = batch_size

        self.trajectories = deque(maxlen=num_trajectories)
        self.current_trajectory: Dict[str, List[torch.Tensor]] = (
            self._init_trajectory_dict()
        )

        # For async batch sampling
        if torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.transfer_stream = None
        self.prefetch_batch: Optional[Dict[str, torch.Tensor]] = None

        # For tracking transition indices
        self.transition_indices: List[Tuple[int, int]] = (
            []
        )  # (trajectory_idx, step_idx)

    def _init_trajectory_dict(self) -> Dict[str, List[torch.Tensor]]:
        """Initialize an empty trajectory dictionary with all required fields"""
        trajectory_dict = {
            "states": [],
            "next_states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        return trajectory_dict

    def _update_transition_indices(self):
        """Update the list of all possible transition indices"""
        self.transition_indices = []
        for traj_idx, trajectory in enumerate(self.trajectories):
            for step_idx in range(trajectory.length):
                self.transition_indices.append((traj_idx, step_idx))

    def add(
        self,
        state: torch.Tensor,
        action: Union[int, torch.Tensor],
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        truncated: bool,
    ) -> None:
        # Convert action to appropriate tensor type
        if self.is_discrete:
            if isinstance(action, torch.Tensor):
                action = action.long()
            else:
                action = torch.tensor(action, dtype=torch.long)
        else:
            if isinstance(action, torch.Tensor):
                action = action.float()
            else:
                action = torch.tensor(action, dtype=torch.float32)

        # Add transition to current trajectory
        self.current_trajectory["states"].append(state)
        self.current_trajectory["next_states"].append(next_state)
        self.current_trajectory["actions"].append(action)
        self.current_trajectory["rewards"].append(torch.tensor(reward))
        self.current_trajectory["dones"].append(torch.tensor(done))

        if done or truncated:
            self._store_current_trajectory()
            self._update_transition_indices()
            self.current_trajectory = self._init_trajectory_dict()

    def _store_current_trajectory(self) -> None:
        if not self.current_trajectory["states"]:
            return

        trajectory = Trajectory(
            states=torch.stack(self.current_trajectory["states"]),
            next_states=torch.stack(self.current_trajectory["next_states"]),
            actions=torch.stack(self.current_trajectory["actions"]),
            rewards=torch.stack(self.current_trajectory["rewards"]),
            dones=torch.stack(self.current_trajectory["dones"]),
            length=len(self.current_trajectory["states"]),
            regression_targets=None,
            advantages=None,
            action_probs=None,
        )
        self.trajectories.append(trajectory)

    def sample_async(self) -> None:
        """Asynchronously prepare next batch of transitions"""
        if self.transfer_stream is None:
            self.prefetch_batch = self._sample_sync()
            return

        with torch.cuda.stream(self.transfer_stream):
            # Sample random transitions
            batch_indices = np.random.choice(
                len(self.transition_indices), size=self.batch_size, replace=True
            )

            states, next_states = [], []
            actions, rewards, dones = [], [], []
            regression_targets, advantages, action_probs = [], [], []

            # Gather transitions
            for idx in batch_indices:
                traj_idx, step_idx = self.transition_indices[idx]
                trajectory = self.trajectories[traj_idx]

                states.append(trajectory.states[step_idx])
                next_states.append(trajectory.next_states[step_idx])
                actions.append(trajectory.actions[step_idx])
                rewards.append(trajectory.rewards[step_idx])
                dones.append(trajectory.dones[step_idx])
                regression_targets.append(trajectory.regression_targets[step_idx])
                advantages.append(trajectory.advantages[step_idx])
                action_probs.append(trajectory.action_probs[step_idx])

            # Convert to tensors and transfer to device
            batch = {
                "states": torch.stack(states).to(self.device, non_blocking=True),
                "next_states": torch.stack(next_states).to(
                    self.device, non_blocking=True
                ),
                "actions": torch.stack(actions).to(
                    self.device, dtype=self.action_dtype, non_blocking=True
                ),
                "rewards": torch.stack(rewards)
                .float()
                .to(self.device, non_blocking=True),
                "dones": torch.stack(dones).float().to(self.device, non_blocking=True),
                "regression_targets": torch.stack(regression_targets).to(
                    self.device, non_blocking=True
                ),
                "advantages": torch.stack(advantages).to(
                    self.device, non_blocking=True
                ),
                "action_probs": torch.stack(action_probs).to(
                    self.device, non_blocking=True
                ),
            }

            self.prefetch_batch = batch

    def _sample_sync(self) -> Dict[str, torch.Tensor]:
        """Synchronously sample a batch of transitions"""
        batch_indices = np.random.choice(
            len(self.transition_indices), size=self.batch_size, replace=True
        )

        states, next_states = [], []
        actions, rewards, dones = [], [], []
        regression_targets, advantages, action_probs = [], [], []

        for idx in batch_indices:
            traj_idx, step_idx = self.transition_indices[idx]
            trajectory = self.trajectories[traj_idx]

            states.append(trajectory.states[step_idx])
            next_states.append(trajectory.next_states[step_idx])
            actions.append(trajectory.actions[step_idx])
            rewards.append(trajectory.rewards[step_idx])
            dones.append(trajectory.dones[step_idx])
            regression_targets.append(trajectory.regression_targets[step_idx])
            advantages.append(trajectory.advantages[step_idx])
            action_probs.append(trajectory.action_probs[step_idx])

        batch = {
            "states": torch.stack(states).to(self.device),
            "next_states": torch.stack(next_states).to(self.device),
            "actions": torch.stack(actions).to(self.device, dtype=self.action_dtype),
            "rewards": torch.stack(rewards).to(self.device),
            "dones": torch.stack(dones).to(self.device),
            "regression_targets": torch.stack(regression_targets).to(self.device),
            "advantages": torch.stack(advantages).to(self.device),
            "action_probs": torch.stack(action_probs).to(self.device),
        }

        for field in self.additional_fields:
            if additional_values[field]:
                batch[field] = torch.stack(additional_values[field]).to(self.device)

        return batch

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get the next batch of transitions"""
        if self.transfer_stream is not None:
            torch.cuda.current_stream().wait_stream(self.transfer_stream)

        batch = self.prefetch_batch

        self.sample_async()
        return batch

    def start_async(self) -> None:
        """Start async batch prefetching"""
        self.sample_async()

    def get_trajectories(self) -> List[Trajectory]:
        """Return all stored trajectories"""
        return list(self.trajectories)

    def empty(self) -> None:
        """Clear all stored trajectories"""
        self.trajectories.clear()
        self.current_trajectory = self._init_trajectory_dict()

    def __len__(self) -> int:
        """Return number of stored trajectories"""
        return len(self.trajectories)

    def get_total_transitions(self) -> int:
        """Return total number of transitions across all trajectories"""
        return sum(traj.length for traj in self.trajectories)
