from random import random as rand, shuffle, randint, choice
import numpy as np
import pandas as pd
import numba as nb
import time



index = np.round(np.linspace(0, 100-1, 76)).astype(np.int64)

print(index)