from random import random as rand, shuffle, randint, choice
import numpy as np
import pandas as pd
import numba as nb
import time



# Create an array of 40 indices
indices = np.arange(40)

# Choose 10 random indices to shuffle
subset_size = 40
subset_indices = np.random.choice(indices, size=subset_size, replace=False)

# Create a shuffled version of the subset
shuffled_subset = np.random.permutation(subset_indices)

# Replace the selected indices in the original array with the shuffled version
indices[np.isin(indices, subset_indices)] = shuffled_subset

print(indices)