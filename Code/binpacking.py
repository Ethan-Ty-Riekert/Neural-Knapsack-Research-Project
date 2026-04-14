"""Bin packing environment implementation and formulation"""
import numpy as np
from typing import List

class Bin:
    """Bin definition for bin packing, representing the physical machine at the datacentre"""
    def __init__(self, capacity: np.ndarray) -> None:
        """Bin initialisation. \\
            Inputs: 
                Capacity: numpy.array of the resource capacity \\
            Outputs: 
                None"""
        self.capacity = capacity.copy()
        self.remaining = capacity.copy()
        self.items = [] # Has to be a dynamic array as items will be added and removed

    def check_fit(self, item: np.ndarray) -> bool:
        """Checks if the item can fit in bin. Includes 0 as being true such that
        the bin can be completely utilized """
        return np.all(self.remaining - item >= 0)
    
    def add(self, item: np.ndarray) -> None:
        """Adds an item to the bin"""
        self.items.append(item)
        self.remaining -= item


class BinPackingEnv:
    """Bin packing environment"""
    def __init__(self, items: list = None, bins:List[Bin]=None) -> None:
        """Inputs:
            """
        self.items = items.copy()
        self.num_items = len(items)
    
        # Bins
        self.bins = []

    def add_bin(self, capacity:np.ndarray) -> None:
        """Add a bin to the bin packing environment"""
        self.bins.append(Bin(capacity))