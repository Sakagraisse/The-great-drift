from random import random as rand, shuffle, randint, choice
import numpy as np
import pandas as pd
import numba as nb
import time

indices = np.arange(5)
print(indices)

np.random.shuffle(indices)

print(indices.shape[0])
print(indices)