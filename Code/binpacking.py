"""Simple bin packing environment implementation and formulation.
Has n dimensions, a bin is defined by its capacity, and the environment takes items one at a time,
and knows all items that will come in and their order."""
import numpy as np
from typing import List, Dict, Union

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


class BasicBinPackingEnv:
    """Basic bin packing environment, items (VM requests) arrive one at a time, all items and their order aleady known,
    bins can be created if action states so."""
    def __init__(self, items: List[np.ndarray], newbinCapacity:np.ndarray, bins:List[Bin]=None) -> None:
        """___"""
        # Items
        self.items = items
        self.num_items = len(items)
        self.item_index = 0 # The index of which item we are dealing with

        # Bins
        self.bins = bins
        self.num_bins = len(bins) if bins is not None else 0
        self.dimensionality = len(newbinCapacity)

        # The capacity of any new bins to be created
        self.new_bin_capacity = newbinCapacity

    def total_items(self) -> int:
        """Return the total amount of items in the environment"""
        return self.num_items
    
    def cur_item(self) -> np.ndarray | None:
        """Return the current item bin environment is looking at"""
        return self.items[self.item_index] if self.item_index < self.num_items else None
    
    def total_bins(self) -> int:
        """Return the total amount of bins in the environment"""
        return self.num_bins

    def reset(self):
        """Reset the environment to the initial state"""
        self.bins = []
        self.num_bins = 0
        self.item_index = 0
        return self.get_state()
    
    def get_state(self) -> Dict[str, Union[np.ndarray, List[np.ndarray], int]]:
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
    
    def get_stateVec(self) -> np.ndarray:
        """
        Return the state as a flat numpy vector:
        [ current_item , bin1_remaining , bin2_remaining , ... ]
        """
        item = self.cur_item()
        if item is None:
            item = np.zeros(self.dimensionality)

        # Flatten bin remaining capacities
        if self.num_bins > 0:
            bin_rem = np.concatenate([b.remaining for b in self.bins])
        else:
            bin_rem = np.array([])

        # Final state vector
        stateVec = np.concatenate([item, bin_rem])

        return stateVec

    
    def step(self, action:int) -> tuple[Dict[str, Union[np.ndarray, int, None]], int, bool]:
        """Place current item into bin index based on provided action.
        Inputs:
            action (int): Bin to place current item"""
        item = self.cur_item()
        
        # Create new bin if decided so
        new_bin_created = False
        if action == self.num_bins:
            self.bins.append(Bin(self.new_bin_capacity))
            self.num_bins += 1
            reward = -10 # For opening a new bin
            new_bin_created = True

        chosen_bin = self.bins[action]

        # Rewards
        if not chosen_bin.check_fit(item):
            reward = -100 # infeasible placement
        elif new_bin_created is False:
            chosen_bin.add(item)
            reward = 0

        # Move to next item
        self.item_index += 1
        # Done if reached all items
        done = self.item_index == self.num_items

        return self.get_state(), reward, done
    
    def get_action_mask(self) -> np.ndarray:
        """
        Returns a binary mask of valid actions.
        Action i is valid if:
        - i < num_bins AND bin i can fit the item
        - OR i == num_bins (create new bin)
        """
        mask = np.zeros(self.num_bins + 1, dtype=np.int32)
        item = self.cur_item()

        # Existing bins
        for i in range(self.num_bins):
            if self.bins[i].check_fit(item):
                mask[i] = 1

        # Creating a new bin is always allowed
        mask[self.num_bins] = 1

        return mask
    



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
    # Visualizing bins
    bins = generate_random_bins(n_bins=6, dim=3)
    plot_bins_3d_prisms(bins)