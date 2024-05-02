from random import random as rand, shuffle, randint, choice
import numpy as np
import pandas as pd
import numba as nb
import time



@nb.jit(nopython=True)
def custom_random_choice(prob):
    # Generate a random number
    rand = np.random.random()
    # Find the index where the cumulative sum exceeds the random number
    for i, val in enumerate(np.cumsum(prob)):
        if rand < val:
            return i
    return len(prob) - 1

@nb.jit(nopython=True)
def return_pop_vector_Ui(value,fitness):
    cumulative = np.empty(24)
    for i in range(24):
        cumulative[i] = np.sum(fitness[0:i])

    sum = np.sum(fitness)
    prob = fitness / sum
    test = np.empty(24, dtype=np.int64)
    for i in range(24):
        test[i] = value[custom_random_choice(prob)]
    return test

value = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
fitness = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

print(return_pop_vector_Ui(value,fitness))



value = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
fitness = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

print(return_pop_vector_Ui(value,fitness))
