"""The gym wrapper for the bin packing environment to allow for RL implementation"""
import numpy as np 
import gymnasium as gym
from typing import List

from Code.binpacking import BasicBinPackingEnv

class GymBinPackingEnv(gym.Env):
    """Gynmasium wrapper for the BasicBinPackingEnv.
    Converts the custom environment into a Gym-compatible one"""

    metadata = {"render_modes": []}

    def __init__(self, items: List[np.ndarray],  bin_capacity:np.ndarray, max_size:int=50) -> None:
        """Gym bin packing environment initialisation"""
        super().__init__() # Have to initialise the parent gym environment

        ### The internal environment
        self.env = BasicBinPackingEnv(items, bin_capacity)

        ### Observation Space
        # State vector = [item_dim + num_bins * dim]
        # As num_bins grows dynamically and we require a fiexed length, we set a max size
        self.item_dim = self.env.dimensionality
        self.max_bins = max_size

        obs_dim = self.item_dim + self.max_bins * self.item_dim

        self.observation_space = gym.spaces.Box(
            low = 0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        ### Action Space
        # Action = chooose bin index to place item or create new bin
        self.action_space = gym.spaces.Discrete(self.max_bins + 1)

    def _pad_state(self, stateVec):
        """Gym required method. Pad the state vector to fixed size for Gym"""
        padded = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        padded[:len(stateVec)] = stateVec
        return padded