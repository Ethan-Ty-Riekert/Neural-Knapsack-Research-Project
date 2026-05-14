"""The gym wrapper for the scheduling environment to allow for RL implementation
https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium
https://gymnasium.farama.org/
"""
import numpy as np 
import gymnasium as gym
from typing import List


class GymSchedulingEnv(gym.Env):
    """Gymnasium wrapper for SchedulingEnv.
    Converts (job, machine) actions into a single integer action.
    Produces a fixed-size observation vector.
    Includes action masking
    """

    metadata = {"render_modes": []} # for gymnasium