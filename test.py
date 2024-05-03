from random import random as rand, shuffle, randint, choice
import numpy as np
import pandas as pd
import numba as nb
import time


for i in range(0,24,2):
    rd_value = np.random.randint(0, 2)
    p1 = i + rd_value
    p2 = i + 1 - rd_value
    print(p1)
    print(p2)