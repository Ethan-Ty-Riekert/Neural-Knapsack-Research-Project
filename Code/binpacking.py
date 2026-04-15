"""Bin packing environment implementation and formulation"""
import numpy as np
from typing import List, Dict

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

    def check_and_add(self, item: np.ndarray) -> None:
        """Checks to see if an item can be added to the bin, and adds it"""
        canfit = self.check_fit(item)
        if canfit is False:
            raise ValueError("Item does not fit in bin")
        
        self.add(item)


class basicBinPackingEnv:
    """Basic bin packing environment, items (VM requests) arrive one at a time, all items and bins are aleady known."""
    def __init__(self, items: List = None, bins:List[Bin] = None, dimensionality:int=0) -> None:
        """___"""
        # Items
        self.items = items
        self.num_items = len(items)
        self.item_index = 0 # The index of which item we are dealing with

        # Bins
        self.bins = bins
        self.num_bins = len(bins)
        if dimensionality == 0 and self.num_bins != 0:
            self.dimensionality = bins[0].capacity.copy()
        else:
            self.dimensionality = dimensionality

    def total_items(self) -> int:
        """Return the total amount of items in the environment"""
        return self.num_items
    
    def total_bins(self) -> int:
        """Return the total amount of bins in the environment"""
        return self.num_bins

    def _add_bins(self, capacityList:List[np.ndarray]) -> None:
        """Add a list of bins to the bin packing environment given a capacity vector.
        Should not be needed in this bin packing where everything is known."""
        for capacity in capacityList:
            self.bins.append(Bin(capacity))
        # Recalculate the size
        self.num_bins = len(self.bins)

    def reset(self):
        """Reset the environment to the initial state"""
        self.bins = []
        self.num_bins = 0
        self.item_index = 0
        return self.get_state()
    
    def get_state(self) -> Dict[str, np.ndarray | int | None]:
        """State representation of the bin packing environment. Returns a dictionary
        of the current item vector and the list of remaining capacities of each bin"""
        capacityArr = np.zeros((self.num_bins, self.dimensionality))
        for n, b in enumerate(self.bins):
            capacityArr[n] = b.remaining

        # Find the current task being placed
        if self.item_index < self.num_items:
            cur_item = self.items[self.item_index]
        else: # We have placed every item
            cur_item = None
    
        return {
            "current_item": cur_item,
            "bin_capacities": capacityArr,
            "num_bins": self.num_bins
        }
    
    def step(self, action):
        """Place current item into bin index based on provided action"""
        ...
        

                    ###### AI GENERATED ########
def generate_random_bins(
    n_bins: int = None,
    dim: int = 2,
    capacity_low: int = 5,
    capacity_high: int = 20
) -> List[Bin]:
    """
    Generate a list of bins with random capacities.
    
    n_bins: number of bins (if None, choose random 1–10)
    dim: dimensionality (1D, 2D, or 3D)
    capacity_low/high: range for random capacities
    """
    if n_bins is None:
        n_bins = np.random.randint(1, 10)

    bins = []
    for _ in range(n_bins):
        cap = np.random.randint(capacity_low, capacity_high, size=dim)
        bins.append(Bin(capacity=cap))

    return bins


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_bins_3d_prisms(bins):
    """
    Plot each bin as a 3D rectangular prism representing its remaining capacity.
    Each bin is placed immediately after the previous one along the X-axis.
    """
    if len(bins) == 0:
        print("No bins to plot.")
        return

    dim = len(bins[0].remaining)
    if dim != 3:
        raise ValueError("3D prism plotting only works for 3-dimensional bins.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(bins)))

    # Track cumulative offset along X
    current_offset = 0.0

    for idx, b in enumerate(bins):
        x, y, z = b.remaining

        # Offset for this bin
        ox = current_offset
        oy = 0
        oz = 0

        # Update offset for next bin
        current_offset += x

        # Vertices of the prism with offset applied
        vertices = np.array([
            [ox,     oy,     oz],
            [ox + x, oy,     oz],
            [ox + x, oy + y, oz],
            [ox,     oy + y, oz],
            [ox,     oy,     oz + z],
            [ox + x, oy,     oz + z],
            [ox + x, oy + y, oz + z],
            [ox,     oy + y, oz + z]
        ])

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]

        poly = Poly3DCollection(faces, alpha=0.25, facecolor=colors[idx])
        poly.set_edgecolor('k')
        ax.add_collection3d(poly)

        ax.text(ox, oy, oz, f"Bin {idx}", color='black')

    ax.set_xlabel("X (stacked bins)")
    ax.set_ylabel("Y (capacity dim 2)")
    ax.set_zlabel("Z (capacity dim 3)")
    ax.set_title("3D Bin Remaining Capacity (Chained Prisms)")

    plt.show()
                ###### END AI GENERATED ########


if __name__ == "__main__":
    # Testing bins


    bins = generate_random_bins(n_bins=6, dim=3)
    plot_bins_3d_prisms(bins)