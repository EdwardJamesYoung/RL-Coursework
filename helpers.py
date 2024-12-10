import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

from jaxtyping import jaxtyped, Integer, Float


from typing import List, Optional

from torch import nn


def get_mlp(
    input_dim: int,
    widths: List[int],
    output_dim: Optional[int],
    non_linearity: str = "ReLU",
):
    non_linearity_fn = getattr(nn, non_linearity)
    layers = []
    prev_width = input_dim
    for width in widths:
        layers.append(nn.Linear(prev_width, width))
        layers.append(non_linearity_fn())
        prev_width = width
    if output_dim is not None:
        layers.append(nn.Linear(prev_width, output_dim))

    return nn.Sequential(*layers)


def visualise(agent, env: gym.Env, n_steps: int):
    frames = []

    print("Generating frames...")
    # ------------ Generate frames ------------
    state, info = env.reset()
    for _ in range(n_steps):
        frames.append(env.render())

        action = agent.evaluation_policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state, info = env.reset()
        else:
            state = next_state

    print("Rendering frames...")
    # ------------ Render frames ------------
    for frame in frames:
        clear_output(wait=True)
        plt.imshow(frame)
        plt.show()
        sleep(0.003)


def evaluate(agent, env: gym.Env, num_episodes: int):
    returns = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_return = 0

        while True:
            action = agent.evaluation_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            episode_return += reward
            if terminated or truncated:
                break

        returns.append(episode_return)
    return returns
