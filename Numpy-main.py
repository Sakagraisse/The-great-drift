from random import random as rand, shuffle, randint, choice
import numpy as np
import pandas as pd
import numba as nb
import time

################
# Create base player
################

#Create numba arrays


#POP creation

@nb.jit(nopython=True)
def create_initial_pop(group_size, number_groups, num_interactions):
    """
    Creates the initial population of players.
    """
    x_i = np.ones((number_groups, group_size), dtype=np.float32)
    d_i = np.ones((number_groups, group_size), dtype=np.float32)
    a_i = np.zeros((number_groups, group_size), dtype=np.float32)
    store_interaction = np.empty((number_groups, group_size,num_interactions), dtype=np.float32)
    endo_fit = np.empty((number_groups, group_size,num_interactions), dtype=np.float32)
    fitness = np.ones((number_groups, group_size), dtype=np.float32)

    return x_i, d_i, a_i, store_interaction, endo_fit, fitness


@nb.jit(nopython=True)
def mutate(value, mu, step_size):
    """
    Applies a mutation to the given value based on the mutation probability mu.
    """
    if value not in {0, 1}:
        if np.random.random() < mu:
            # Mutation: decide the step direction (up or down)
            step_direction = np.random.choice(np.array([-step_size, step_size]))
            # Apply the mutation while staying within the [0,1] boundaries
            new_value = min(1, max(0, value + step_direction))
        else:
            # No mutation, the value remains the same
            new_value = value
    else:
        # The value is at the boundary of the grid, it can only move in one direction
        if value == 0 and np.random.random() < mu/2:
            new_value = value + step_size
        elif value == 1 and np.random.random() < mu/2:
            new_value = value - step_size
        else:
            # The value remains the same
            new_value = value

    return new_value
def meta_pop(pop):
    """
    Create the movement of pop groups
    """
    pass

@nb.jit(nopython=True)
def migration(x_i, d_i, a_i, store_interaction, endo_fit, fitness, to_migrate,number_groups, group_size):
    """
    Create the migration of pop groups
    """
    temp_x_i = np.empty((number_groups,to_migrate), dtype=np.float32)
    temp_d_i = np.empty((number_groups,to_migrate), dtype=np.float32)
    temp_a_i = np.empty((number_groups,to_migrate), dtype=np.float32)

    # extract from the pop the migrtantes
    for i in range(number_groups):
        indices = np.arange(x_i.shape[1])
        np.random.shuffle(indices)
        x_i[i, :] = x_i[i, indices]
        d_i[i, :] = d_i[i, indices]
        a_i[i, :] = a_i[i, indices]

        temp_x_i[i,:] = x_i[i,0:to_migrate]
        x_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float32)
        temp_d_i[i,:] = d_i[i,0:to_migrate]
        d_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float32)
        temp_a_i[i,:] = a_i[i,0:to_migrate]
        a_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float32)

    temp_x_i = temp_x_i.flatten()
    temp_d_i = temp_d_i.flatten()
    temp_a_i = temp_a_i.flatten()



    for j in range(number_groups):
        x_i[j,0:to_migrate] = temp_x_i[j*to_migrate:(j+1)*to_migrate]
        d_i[j,0:to_migrate] = temp_d_i[j*to_migrate:(j+1)*to_migrate]
        a_i[j,0:to_migrate] = temp_a_i[j*to_migrate:(j+1)*to_migrate]

    return x_i, d_i, a_i


@nb.jit(nopython=True)
def social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitness, number_groups, group_size, num_interactions, transfert_multiplier, truc):
    """
    Create the social dilemma
    """
    for j in range(number_groups):
        for i in range(0, group_size, 2):
            # select the first player
            rd_value = np.random.randint(0, 2)
            p1 = i + rd_value
            p2 = i + 1 - rd_value
            for k in range(num_interactions):
                if i == 0:
                    store_interaction[j, p1, k] = x_i[j, p1]
                    store_interaction[j, p2, k] = a_i[j, p2] + (d_i[j, p2] - a_i[j, p2]) * x_i[j, p1]

                else:
                    store_interaction[j, p1, k] = a_i[j, p1] + (d_i[j, p1] - a_i[j, p2]) * store_interaction[j, p2, k-1]
                    store_interaction[j, p2, k] = a_i[j, p2] + (d_i[j, p2] - a_i[j, p2]) * store_interaction[j, p1, k]
                endo_fit[j, p1, k] = 1 - store_interaction[j, p1, k] + store_interaction[j, p2, k] * transfert_multiplier
                endo_fit[j, p2, k] = 1 - store_interaction[j, p2, k] + store_interaction[j, p1, k] * transfert_multiplier

            fitness[j, p1] = (1-truc)*num_interactions + truc * np.sum(endo_fit[j, p1, :])
            fitness[j, p2] = (1-truc)*num_interactions + truc * np.sum(endo_fit[j, p2, :])

    return x_i, d_i, a_i, store_interaction, endo_fit, fitness

@nb.jit(nopython=True)
def reproduction(x_i, d_i, a_i, fitness, mu, step_size):
    """
    Reproduction of the population
    """
    for j in range(40):
            #for x_i
            x_i[j, :] = return_pop_vector_Ui(x_i[j, :], fitness[j, :])
            #for d_i
            d_i[j, :] = return_pop_vector_Ui(d_i[j, :], fitness[j, :])
            #for a_i
            a_i[j, :] = return_pop_vector_Ui(a_i[j, :], fitness[j, :])

    return x_i, d_i, a_i


@nb.jit(nopython=True)
def main_loop(period, transfert_multiplier, frame_a, frame_x, frame_d, mu, step_size, to_migrate, number_of_interaction, truc, group_size, number_groups, num_interactions):
    # create the initial population
    x_i, d_i, a_i, store_interaction, endo_fit, fitness = create_initial_pop(group_size, number_groups, num_interactions)

    #migration
    x_i, d_i, a_i = migration(x_i, d_i, a_i, store_interaction, endo_fit, fitness, to_migrate, number_groups, group_size)

    #social dilemna
    x_i, d_i, a_i, store_interaction, endo_fit, fitness = social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitness, number_groups, group_size, num_interactions, transfert_multiplier, truc)

    store_data(x_i, d_i, a_i, frame_a, frame_x, frame_d, 0)
    #reproduction
    x_i, d_i, a_i = reproduction(x_i, d_i, a_i, fitness, mu, step_size)

    #main loop
    for i in range(1, period, 1):
        print(i)
        # migration
        x_i, d_i, a_i = migration(x_i, d_i, a_i, store_interaction, endo_fit, fitness, to_migrate, number_groups, group_size)
        # social dilemna
        x_i, d_i, a_i, store_interaction, endo_fit, fitness = social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitness, number_groups, group_size, num_interactions, transfert_multiplier, truc)
        # store the data
        store_data(x_i, d_i, a_i, frame_a, frame_x, frame_d, i)
        # reproduction
        #x_i, d_i, a_i = reproduction(x_i, d_i, a_i, fitness, mu, step_size)

    return x_i, d_i, a_i, fitness

@nb.jit(nopython=True)
def store_data(x_i, d_i, a_i, frame_a, frame_x, frame_d, period):
    """
    Store the data in a numpy array
    """
    counter = 0
    # iterate on every pop number
    for j in range(40):
        # iterate on every player in the population
        for i in range(24):
            # store the data in the numpy arrays
            frame_a[counter, period] = a_i[j, i]
            frame_x[counter, period] = x_i[j, i]
            frame_d[counter, period] = d_i[j, i]
            counter += 1

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



# Définition des valeurs et des probabilités
#values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
#probabilities = np.array([0.04] * 24)  # Exemple avec des probabilités uniformes
#probabilities /= probabilities.sum()  # Assurez-vous que la somme est 1

# base parameters

group_size = 24
number_groups = 40
num_interactions = 10
number_of_interaction = num_interactions
to_migrate = 8

period = 150000
mu = 0.2
step_size = 0.025
truc = 0.5

frame_a = np.zeros((960, period))
frame_x = np.zeros((960, period))
frame_d = np.zeros((960, period))

#run the main loop
start = time.time()
x_i, d_i, a_i, fitness = main_loop(period, 2, frame_a, frame_x, frame_d, mu, step_size, to_migrate, number_of_interaction, truc, group_size, number_groups, num_interactions)
end = time.time()

print(fitness)
print("Execution time: ", end - start,"for", period, "iterations.")


start = time.time()

np.save('frame_a.npy', frame_a)
np.save('frame_x.npy', frame_x)
np.save('frame_d.npy', frame_d)

end = time.time()
print("Saving time: ", end - start)


