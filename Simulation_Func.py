import sys
import os
import numpy as np
import numba as nb
import time

################
# Create base player
################

@nb.jit(nopython=True)
def create_frames(period, group_size, number_groups):
    # Initialize the frames
    frame_a = np.zeros((period, (group_size * number_groups)))
    frame_x = np.zeros((period, (group_size * number_groups)))
    frame_d = np.zeros((period, (group_size * number_groups)))
    frame_t = np.zeros((period, (group_size * number_groups)))
    frame_u = np.zeros((period, (group_size * number_groups)))
    frame_v = np.zeros((period, (group_size * number_groups)))
    frame_fitnessToT = np.zeros((period, (group_size * number_groups)))

    # Initialize the index array
    index = np.round(np.linspace(0, frame_x.shape[0]-1, 75)).astype(np.int64)
    print(index)

    return frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT, index


@nb.jit(nopython=True)
def create_initial_pop(group_size, number_groups, num_interactions):
    """
    This function creates the initial population of players for a simulation.

    Parameters:
    group_size (int): The size of each group in the population.
    number_groups (int): The total number of groups in the population.
    num_interactions (int): The number of interactions each player has.
    """
    x_i = np.ones((number_groups, group_size), dtype=np.float64)
    d_i = np.ones((number_groups, group_size), dtype=np.float64)
    a_i = np.zeros((number_groups, group_size), dtype=np.float64)
    t_i = np.zeros((number_groups, group_size), dtype=np.float64)
    u_i = np.zeros((number_groups, group_size), dtype=np.float64)
    v_j = np.ones((number_groups, group_size), dtype=np.float64)

    store_interaction = np.zeros((number_groups, group_size,num_interactions), dtype=np.float64)
    fitnessIN = np.zeros((number_groups, group_size), dtype=np.float64)
    fitnessOUT = np.zeros((number_groups, group_size), dtype=np.float64)
    fitnessToT = np.zeros((number_groups, group_size), dtype=np.float64)

    return x_i, d_i, a_i, t_i, u_i, v_j, store_interaction, fitnessIN, fitnessOUT, fitnessToT


@nb.jit(nopython=True)
def store_data(x_i, d_i, a_i, t_i, u_i , v_i,fitnessToT , frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT, period):
    """
    Store the data in numpy arrays for a given period.

    Parameters: x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT, period

    Returns: frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT
    """
    frame_a[period, :] = a_i.flatten()
    frame_x[period, :] = x_i.flatten()
    frame_d[period, :] = d_i.flatten()
    frame_t[period, :] = t_i.flatten()
    frame_u[period, :] = u_i.flatten()
    frame_v[period, :] = v_i.flatten()
    frame_fitnessToT[period, :] = fitnessToT.flatten()
    return frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT

@nb.jit(nopython=True)
def setdiff1d_numba(arr1, arr2):
    """
        Compute the set difference of two arrays using Numba. The set difference
        returns the elements that are in arr1 but not in arr2.

        Parameters: arr1, arr2

        Returns: numpy.ndarray
    """
    return np.asarray(list(set(arr1) - set(arr2)))

@nb.jit(nopython=True)
def meta_pop(number_groups, frame_x, frame_a, frame_d):
    """
    Form the meta-population for out-group interactions by randomly shuffling and selecting groups.

    Parameters: number_groups, frame_x, frame_a, frame_d

    Returns: frame_x, frame_a, frame_d

    not used here
    """
    indices = np.arange(40)
    #generate a random number between 0, 20 , 40
    to_move = np.random.choice(np.arange(10, number_groups+1, 20))
    #generate a vector of to_move elements with random ints between 0 and 40
    subset_indices = np.random.choice(indices, size=to_move, replace=False)
    # Get the remaining indices
    shuffled_subset = np.random.permutation(subset_indices)
    # Replace the selected indices in the original array with the shuffled version
    mask = np.array([i in subset_indices for i in indices])
    indices[np.where(mask)] = shuffled_subset
    frame_x = frame_x[indices, :]
    frame_a = frame_a[indices, :]
    frame_d = frame_d[indices, :]
    return frame_x, frame_a, frame_d


@nb.jit(nopython=True)
def migration(x_i, d_i, a_i,t_i,u_i,v_i, to_migrate,number_groups, group_size):
    """
    Create the migration of pop groups
    """
    if to_migrate > group_size:
        raise ValueError("The number of migrants is greater than the group size")

    if to_migrate == 0:
        return x_i, d_i, a_i, t_i, u_i, v_i

    temp_x_i = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_d_i = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_a_i = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_t_i = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_u_i = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_v_i = np.zeros((number_groups,to_migrate), dtype=np.float64)

    # extract from the pop the migrants
    for i in range(number_groups):
        indices = np.arange(group_size)
        np.random.shuffle(indices)
        x_i[i,:] = x_i[i,indices]
        d_i[i,:] = d_i[i,indices]
        a_i[i,:] = a_i[i,indices]
        t_i[i,:] = t_i[i,indices]
        u_i[i,:] = u_i[i,indices]
        v_i[i,:] = v_i[i,indices]
        temp_x_i[i,:] = x_i[i,0:to_migrate]
        x_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_d_i[i,:] = d_i[i,0:to_migrate]
        d_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_a_i[i,:] = a_i[i,0:to_migrate]
        a_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_t_i[i,:] = t_i[i,0:to_migrate]
        t_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_u_i[i,:] = u_i[i,0:to_migrate]
        u_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_v_i[i,:] = v_i[i,0:to_migrate]
        v_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)

    temp_x_i = temp_x_i.ravel()
    temp_d_i = temp_d_i.ravel()
    temp_a_i = temp_a_i.ravel()
    temp_t_i = temp_t_i.ravel()
    temp_u_i = temp_u_i.ravel()
    temp_v_i = temp_v_i.ravel()

    indices2 = np.arange((to_migrate * number_groups))
    np.random.shuffle(indices2)

    temp_x_i[:] = temp_x_i[indices2]
    temp_d_i[:] = temp_d_i[indices2]
    temp_a_i[:] = temp_a_i[indices2]
    temp_t_i[:] = temp_t_i[indices2]
    temp_u_i[:] = temp_u_i[indices2]
    temp_v_i[:] = temp_v_i[indices2]

    for j in range(number_groups):
        x_i[j,0:to_migrate] = temp_x_i[j*to_migrate:(j+1)*to_migrate]
        d_i[j,0:to_migrate] = temp_d_i[j*to_migrate:(j+1)*to_migrate]
        a_i[j,0:to_migrate] = temp_a_i[j*to_migrate:(j+1)*to_migrate]
        t_i[j,0:to_migrate] = temp_t_i[j*to_migrate:(j+1)*to_migrate]
        u_i[j,0:to_migrate] = temp_u_i[j*to_migrate:(j+1)*to_migrate]
        v_i[j,0:to_migrate] = temp_v_i[j*to_migrate:(j+1)*to_migrate]
    return x_i, d_i, a_i, t_i, u_i, v_i




@nb.jit(nopython=True)
def IN_social_dilemma(x_i, d_i, a_i, store_interaction, fitnessIN, number_groups, group_size, num_interactions, transfert_multiplier):
    """
    Create the social dilemma
    """
    for j in range(number_groups):
        indices = np.arange(group_size)
        np.random.shuffle(indices)
        x_i[j,:] = x_i[j,indices]
        d_i[j,:] = d_i[j,indices]
        a_i[j,:] = a_i[j,indices]
        for i in range(0, group_size, 2):
            p1 = i
            p2 = i + 1
            store_interaction[j, p1, 0] = x_i[j, p1]
            store_interaction[j, p2, 0] = a_i[j, p2] + (d_i[j, p2] - a_i[j, p2]) * \
                                               store_interaction[j, p1, 0]

            fitnessIN[j, p1] = 1 - store_interaction[j, p1, 0] + store_interaction[j, p2, 0] * transfert_multiplier
            fitnessIN[j, p2] = 1 - store_interaction[j, p2, 0] + store_interaction[j, p1, 0] * transfert_multiplier

            if num_interactions > 1:
                for k in range(1,num_interactions):
                    store_interaction[j, p1, k] = a_i[j, p1] + (d_i[j, p1] - a_i[j, p1]) * store_interaction[j, p2, k-1]
                    store_interaction[j, p2, k] = a_i[j, p2] + (d_i[j, p2] - a_i[j, p2]) * store_interaction[j, p1, k]

                    fitnessIN[j, p1] += 1 - store_interaction[j, p1, k] + store_interaction[j, p2, k] * transfert_multiplier
                    fitnessIN[j, p2] += 1 - store_interaction[j, p2, k] + store_interaction[j, p1, k] * transfert_multiplier

    return x_i, d_i, a_i, store_interaction, fitnessIN


@nb.jit(nopython=True)
def fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, number_groups, group_size,truc,num_interactions,compi = 0):
    """
    Calculate the total fitness
    """
    for j in range(number_groups):
        for i in range(group_size):
            fitnessToT[j,i] = (1-truc)*(num_interactions + compi) + truc *(fitnessIN[j,i] + fitnessOUT[j,i])
    return fitnessToT


@nb.jit(nopython=True)
def mutate(value, mu, step_size):
    """
    Applies a mutation to the given value based on the mutation probability mu.
    """
    if value < 0 or value > 1:
        raise ValueError("The value must be within the range [0, 1].")
    if mu < 0 or mu > 1:
        raise ValueError("The mutation probability mu must be within the range [0, 1].")
    if step_size <= 0 or step_size > 1:
        raise ValueError("The step size must be within the range (0, 1].")

    if value > 0 and value < 1:
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
        if value == 0 and np.random.random() < mu / 2:
            new_value = value + step_size
        elif value == 1 and np.random.random() < mu / 2:
            new_value = value - step_size
        else:
            # The value remains the same
            new_value = value
    return new_value

@nb.jit(nopython=True)
def reproduction_pop(v1,v2,v3, fitnessToT,number_groups, mu, step_size):
    """
    Reproduction of the population
    """
    for j in range(number_groups):
        v1[j,:], v2[j,:], v3[j,:] = reproduction_one_group(v1[j,:], v2[j,:], v3[j,:], fitnessToT[j,:], mu, step_size)
    return v1, v2, v3


@nb.jit(nopython=True)
def reproduction_one_group(v1,v2,v3, fitnessToT, mu, step_size):
    """
    Reproduction of one group of the population
    """
    #for x_i
    v1 = return_pop_vector_Ui(v1, fitnessToT)
    #for d_i
    v2 = return_pop_vector_Ui(v2, fitnessToT)
    #for a_i
    v3 = return_pop_vector_Ui(v3, fitnessToT)
    for i in range(v1.shape[0]):
        v1[i] = mutate(v1[i], mu, step_size)
        v2[i] = mutate(v2[i], mu, step_size)
        v3[i] = mutate(v3[i], mu, step_size)
    return v1, v2, v3


@nb.jit(nopython=True)
def custom_random_choice(prob):
    # Generate a random number
    rand = np.random.random()
    cum_prob = np.cumsum(prob)
    # Find the index where the cumulative sum exceeds the random number
    for i in range(len(cum_prob)):
        if rand < cum_prob[i]:
            return i
    return len(prob) - 1

@nb.jit(nopython=True)
def return_pop_vector_Ui(value,fitness):
    total_group_fitness = np.sum(fitness)
    prob = fitness / total_group_fitness
    test_size = value.shape[0]
    test = np.empty(test_size, dtype=np.float64)
    for i in range(test_size):
        test[i] = value[custom_random_choice(prob)]

    return test

@nb.jit(nopython=True)
def costum_shuffle_pop(x_i,a_i,d_i,fitness):
    indices = np.arange(x_i.shape[0])
    np.random.shuffle(indices)
    x_i = x_i[indices]
    d_i = d_i[indices]
    a_i = a_i[indices]
    fitness = fitness[indices]

    return x_i, a_i, d_i, fitness



#@nb.jit(nopython=True)
def main_loop_iterated(x_i, d_i, a_i, t_i, u_i, v_i, fitnessIN, fitnessOUT, fitnessToT,store_interaction, \
                       frame_a, frame_x, frame_d, frame_t,frame_u, frame_v, frame_fitnessToT,\
                        group_size, number_groups, num_interactions, period,mu, step_size,coupled, to_migrate, transfert_multiplier, truc, tracking):

    compi=0
    for i in range(0, period, 1):
        #print("Progress:", round((i / period) * 100), "%")
        # store the data
        frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT = store_data(x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, \
                   frame_fitnessToT, i)

        #Migration(Coupled)
        if coupled:
            x_i, d_i, a_i, t_i, u_i, v_i = migration(x_i, d_i, a_i, t_i, u_i, v_i, to_migrate, number_groups, group_size)

        #Social Dilemma
        x_i, d_i, a_i, store_interaction, fitnessIN = IN_social_dilemma(x_i, d_i, a_i, store_interaction, fitnessIN, number_groups, group_size,\
                                                                    num_interactions, transfert_multiplier)

        fitnessToT = fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, number_groups, group_size, truc, num_interactions,
                               compi)

        #Migration (Decoupled)
        if not coupled:
             x_i, d_i, a_i, t_i, u_i, v_i = migration(x_i, d_i, a_i, t_i, u_i, v_i, to_migrate, number_groups, group_size)

        #Reproduction
        x_i, d_i, a_i = reproduction_pop(x_i, d_i, a_i, fitnessToT,number_groups, mu, step_size)

        tracking[0] = i/period

    return frame_a, frame_x, frame_d


#@nb.jit(nopython=True)
def loop_iterated(group_size, number_groups, num_interactions, period,mu, step_size,coupled, to_migrate, transfert_multiplier, truc,\
                    to_average,tracking):


    for i in range(1, (to_average + 1), 1):
        frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT, index = create_frames(period,
                                                                                                      group_size,
                                                                                                      number_groups)

        x_i, d_i, a_i, t_i, u_i, v_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT \
            = create_initial_pop(group_size, number_groups, num_interactions)

        frame_a, frame_x, frame_d = main_loop_iterated(x_i, d_i, a_i, t_i, u_i, v_i, fitnessIN, fitnessOUT, fitnessToT, store_interaction, \
                           frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT, \
                           group_size, number_groups, num_interactions, period, mu, step_size, coupled, to_migrate,\
                           transfert_multiplier, truc,tracking)

        if i == 1:
            frame_x_store = frame_x[index, :]
            frame_a_store = frame_a[index, :]
            frame_d_store = frame_d[index, :]
        if i > 1:
            frame_x_store = np.hstack((frame_x_store, frame_x[index, :]))
            frame_a_store = np.hstack((frame_a_store, frame_a[index, :]))
            frame_d_store = np.hstack((frame_d_store, frame_d[index, :]))
            print(frame_a_store.shape, frame_x_store.shape, frame_d_store.shape)
        tracking[1]= i/ to_average

    return frame_x_store, frame_a_store, frame_d_store


def launch_sim_iterated(group_size, number_groups, num_interactions, period, mu, step_size, \
                                coupled, to_migrate, transfert_multiplier, truc,to_average,tracking):


    dir_path = os.path.dirname(os.path.abspath(__file__))

    frame_x_store, frame_a_store, frame_d_store = loop_iterated(\
                    group_size, number_groups, num_interactions, period,mu, step_size,coupled, to_migrate, transfert_multiplier, truc,\
                    to_average,tracking)

    for file in os.listdir(dir_path):
        if file.endswith(".npy"):
            os.remove(file)

    np.save(os.path.join(dir_path, 'frame_a.npy'), frame_a_store)
    np.save(os.path.join(dir_path, 'frame_x.npy'), frame_x_store)
    np.save(os.path.join(dir_path, 'frame_d.npy'), frame_d_store)

    print("finished")