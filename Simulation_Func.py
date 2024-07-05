import os
import numpy as np
import numba as nb
from math import sqrt


@nb.jit(nopython=True)
def create_frames(period, group_size, number_groups):
    """
        This function creates the arrays to store the result of the simulation.

        Parameters:
        period (int): The number of periods in the simulation.
        group_size (int): The size of each group in the population.
        number_groups (int): The total number of groups in the population.
        """
    # Initialize the frames
    frame_a = np.zeros((period, (group_size * number_groups)))
    frame_x = np.zeros((period, (group_size * number_groups)))
    frame_d = np.zeros((period, (group_size * number_groups)))
    frame_surplus = np.zeros((period, (group_size * number_groups)))
    frame_fitnessToT = np.zeros((period, (group_size * number_groups)))

    # Initialize the index to store the data
    index = np.round(np.linspace(0, frame_x.shape[0]-1, 76)).astype(np.int64)

    return frame_a, frame_x, frame_d,frame_fitnessToT, frame_surplus,index


@nb.jit(nopython=True)
def create_initial_pop(group_size, number_groups, num_interactions,transfert_multiplier,x_i_value,choice):
    """
    This function creates the initial population of players for a simulation.

    Parameters:
    group_size (int): The size of each group in the population.
    number_groups (int): The total number of groups in the population.
    num_interactions (int): The number of interactions each player has.
    """


    # Initialize the population depending on the choice of population type
    # choice = 0: perfect reciprocators
    # choice = 1: unconditionally selfish
    # choice = 2: equilibrium degree of escalation
    if choice == 0:
        x_i = np.ones((number_groups, group_size), dtype=np.float64)*x_i_value
        a_i = np.zeros((number_groups, group_size), dtype=np.float64)
        d_i = np.ones((number_groups, group_size), dtype=np.float64)

    elif choice == 1:
        x_i = np.ones((number_groups, group_size), dtype=np.float64)*x_i_value
        a_i = np.zeros((number_groups, group_size), dtype=np.float64)
        d_i = np.zeros((number_groups, group_size), dtype=np.float64)
    elif choice == 2:
        if num_interactions == 1:
            print("The equilibrium degree is not defined for one interaction")
            a_hat = 1
        else:
            delta = 1 - (1 / num_interactions)
            equilibrium_degree = (delta * (1 - transfert_multiplier) + sqrt(
                (delta ** 2) * ((1 - transfert_multiplier) ** 2) + 4 * transfert_multiplier * delta)) / (
                                         2 * transfert_multiplier * delta)
            a_hat = 1 - equilibrium_degree


        x_i = np.ones((number_groups, group_size), dtype=np.float64)*x_i_value
        #a_i is a_hat
        a_i = np.ones((number_groups, group_size), dtype=np.float64) * a_hat
        d_i = np.ones((number_groups, group_size), dtype=np.float64)
    else:
        raise ValueError("The choice must be 0, 1 or 2")


    # Initialize the store_interaction, fitnessIN, fitnessOUT, fitnessToT, and surplus arrays
    store_interaction = np.zeros((number_groups, group_size,num_interactions), dtype=np.float64)
    surplus = np.zeros((number_groups, group_size), dtype=np.float64)
    fitnessIN = np.zeros((number_groups, group_size), dtype=np.float64)
    fitnessOUT = np.zeros((number_groups, group_size), dtype=np.float64)
    fitnessToT = np.zeros((number_groups, group_size), dtype=np.float64)
    return x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT,surplus


@nb.jit(nopython=True)
def store_data(x_i, d_i, a_i,fitnessToT ,surplus, frame_a, frame_x, frame_d, frame_fitnessToT, frame_surplus,period):
    """
    Store the data in numpy arrays for a given period.

    Parameters:
    x_i : first move
    a_i : left intercept
    d_i : right intercept
    fitnessToT : total fitness
    frame_a, frame_x, frame_d, frame_fitnessToT : numpy.ndarray to store the results
    """

    frame_a[period, :] = a_i.flatten()
    frame_x[period, :] = x_i.flatten()
    frame_d[period, :] = d_i.flatten()
    frame_fitnessToT[period, :] = fitnessToT.flatten()
    frame_surplus[period, :] = surplus.flatten()
    return frame_a, frame_x, frame_d, frame_fitnessToT,frame_surplus

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
def migration(x_i, d_i, a_i, fitnessToT, fitnessOUT, fitnessIN, surplus, number_groups, group_size, to_migrate):

    """
    Migrate the players between groups

    Parameters:
    x_i : first move
    d_i : right intercept
    a_i : left intercept
    fitnessToT : total fitness
    fitnessOUT : fitness out group interaction (place holder for future use)
    fitnessIN : fitness in group interaction
    surplus : surplus of the players
    number_groups : number of groups
    group_size : size of the groups
    to_migrate : number of players to migrate
    """
    if to_migrate > group_size:
        raise ValueError("The number of migrants is greater than the group size")

    # If no migration is needed, return the original population
    if to_migrate == 0:
        return x_i, d_i, a_i, fitnessToT, fitnessOUT, fitnessIN, surplus

    # Create a temporary array to store the migrants
    temp_x_i = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_d_i = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_a_i = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_fitnessToT = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_fitnessOUT = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_fitnessIN = np.zeros((number_groups,to_migrate), dtype=np.float64)
    temp_surplus = np.zeros((number_groups,to_migrate), dtype=np.float64)


    # extract the migrants from the population
    for i in range(number_groups):
        indices = np.arange(group_size)
        np.random.shuffle(indices)
        x_i[i,:] = x_i[i,indices]
        d_i[i,:] = d_i[i,indices]
        a_i[i,:] = a_i[i,indices]
        fitnessToT[i,:] = fitnessToT[i,indices]
        fitnessOUT[i,:] = fitnessOUT[i,indices]
        fitnessIN[i,:] = fitnessIN[i,indices]
        surplus[i,:] = surplus[i,indices]
        temp_x_i[i,:] = x_i[i,0:to_migrate]
        x_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_d_i[i,:] = d_i[i,0:to_migrate]
        d_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_a_i[i,:] = a_i[i,0:to_migrate]
        a_i[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_fitnessToT[i,:] = fitnessToT[i,0:to_migrate]
        fitnessToT[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_fitnessOUT[i,:] = fitnessOUT[i,0:to_migrate]
        fitnessOUT[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_fitnessIN[i,:] = fitnessIN[i,0:to_migrate]
        fitnessIN[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)
        temp_surplus[i,:] = surplus[i,0:to_migrate]
        surplus[i,0:to_migrate] = np.zeros(to_migrate, dtype=np.float64)

    # Shuffle the migrants
    temp_x_i = temp_x_i.ravel()
    temp_d_i = temp_d_i.ravel()
    temp_a_i = temp_a_i.ravel()
    temp_fitnessToT = temp_fitnessToT.ravel()
    temp_fitnessOUT = temp_fitnessOUT.ravel()
    temp_fitnessIN = temp_fitnessIN.ravel()
    temp_surplus = temp_surplus.ravel()
    indices2 = np.arange((to_migrate * number_groups))
    np.random.shuffle(indices2)

    # Reorder the migrants
    temp_x_i[:] = temp_x_i[indices2]
    temp_d_i[:] = temp_d_i[indices2]
    temp_a_i[:] = temp_a_i[indices2]
    temp_fitnessToT[:] = temp_fitnessToT[indices2]
    temp_fitnessOUT[:] = temp_fitnessOUT[indices2]
    temp_fitnessIN[:] = temp_fitnessIN[indices2]
    temp_surplus[:] = temp_surplus[indices2]

    # Migrate the players
    for j in range(number_groups):
        x_i[j,0:to_migrate] = temp_x_i[j*to_migrate:(j+1)*to_migrate]
        d_i[j,0:to_migrate] = temp_d_i[j*to_migrate:(j+1)*to_migrate]
        a_i[j,0:to_migrate] = temp_a_i[j*to_migrate:(j+1)*to_migrate]
        fitnessToT[j,0:to_migrate] = temp_fitnessToT[j*to_migrate:(j+1)*to_migrate]
        fitnessOUT[j,0:to_migrate] = temp_fitnessOUT[j*to_migrate:(j+1)*to_migrate]
        fitnessIN[j,0:to_migrate] = temp_fitnessIN[j*to_migrate:(j+1)*to_migrate]
        surplus[j,0:to_migrate] = temp_surplus[j*to_migrate:(j+1)*to_migrate]

    return x_i, d_i, a_i, fitnessToT, fitnessOUT, fitnessIN, surplus




@nb.jit(nopython=True)
def IN_social_dilemma(x_i, d_i, a_i, store_interaction, fitnessIN, number_groups, group_size, num_interactions, transfert_multiplier,surplus):
    """
    Perform the in-group social dilemma

    Parameters:
    x_i : first move
    d_i : right intercept
    a_i : left intercept
    store_interaction : store the interaction
    fitnessIN : fitness in group interaction
    number_groups : number of groups
    group_size : size of the groups
    num_interactions : number of interactions
    transfert_multiplier : transfer multiplier
    surplus : surplus of the players
    """

    #iterate over the groups
    for j in range(number_groups):
        indices = np.arange(group_size)
        np.random.shuffle(indices)
        x_i[j,:] = x_i[j,indices]
        d_i[j,:] = d_i[j,indices]
        a_i[j,:] = a_i[j,indices]
        #iterate over the players
        for i in range(0, group_size, 2):
            #player 1 and player 2
            p1 = i
            p2 = i + 1
            store_interaction[j, p1, 0] = x_i[j, p1]
            store_interaction[j, p2, 0] = a_i[j, p2] + (d_i[j, p2] - a_i[j, p2]) * \
                                               store_interaction[j, p1, 0]

            fitnessIN[j, p1] = 1 - store_interaction[j, p1, 0] + store_interaction[j, p2, 0] * transfert_multiplier
            fitnessIN[j, p2] = 1 - store_interaction[j, p2, 0] + store_interaction[j, p1, 0] * transfert_multiplier

            #iterate over the interactions
            if num_interactions > 1:
                for k in range(1,num_interactions):
                    store_interaction[j, p1, k] = a_i[j, p1] + (d_i[j, p1] - a_i[j, p1]) * store_interaction[j, p2, k-1]
                    store_interaction[j, p2, k] = a_i[j, p2] + (d_i[j, p2] - a_i[j, p2]) * store_interaction[j, p1, k]

                    fitnessIN[j, p1] += 1 - store_interaction[j, p1, k] + store_interaction[j, p2, k] * transfert_multiplier
                    fitnessIN[j, p2] += 1 - store_interaction[j, p2, k] + store_interaction[j, p1, k] * transfert_multiplier

            #calculate the surplus
            surplus[j, p1] = (np.sum(store_interaction[j, p1, :])*transfert_multiplier)/ num_interactions
            surplus[j, p2] = (np.sum(store_interaction[j, p2, :])*transfert_multiplier)/ num_interactions

    return x_i, d_i, a_i, store_interaction, fitnessIN,surplus


@nb.jit(nopython=True)
def fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, number_groups, group_size,truc,num_interactions):
    """
    Calculate the total fitness

    Parameters:
    fitnessIN : fitness in group interaction
    fitnessOUT : fitness out group interaction
    fitnessToT : total fitness
    number_groups : number of groups
    group_size : size of the groups
    truc : truc parameter
    num_interactions : number of interactions
    """
    for j in range(number_groups):
        for i in range(group_size):
            fitnessToT[j,i] = (1-truc)*(num_interactions) + truc *(fitnessIN[j,i] + fitnessOUT[j,i])
    return fitnessToT


@nb.jit(nopython=True)
def mutate(value, mu, step_size):
    """
    Applies a mutation to the given value based on the mutation probability mu.

    Parameters:
    value : The value to mutate.
    mu : The mutation probability.
    step_size : The step size of the mutation.
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
    Takes three vectors and reproduces them based on the fitness of the group

    Parameters:
    v1 : first vector of parameter
    v2 : second vector of parameter
    v3 : third vector of parameter
    fitnessToT : total fitness
    number_groups : number of groups
    mu : mutation probability
    step_size : step size of the mutation
    """
    for j in range(number_groups):
        v1[j,:], v2[j,:], v3[j,:] = reproduction_one_group(v1[j,:], v2[j,:], v3[j,:], fitnessToT[j,:], mu, step_size)
    return v1, v2, v3


@nb.jit(nopython=True)
def reproduction_one_group(v1,v2,v3, fitnessToT, mu, step_size):
    """
    Reproduction of one group of the population

    Parameters:
    v1 : first vector of parameter
    v2 : second vector of parameter
    v3 : third vector of parameter
    fitnessToT : total fitness
    mu : mutation probability
    step_size : step size of the mutation
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
    """
    Custom random choice function that selects an index based on the given probabilities.
    Args:
        prob:
    """
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
    """
    Return a vector of the population based on the fitness of the group
    Args:
        value
        fitness
    """
    total_group_fitness = np.sum(fitness)
    prob = fitness / total_group_fitness
    test_size = value.shape[0]
    test = np.empty(test_size, dtype=np.float64)
    for i in range(test_size):
        test[i] = value[custom_random_choice(prob)]

    return test

@nb.jit(nopython=True)
def costum_shuffle_pop(x_i,a_i,d_i,fitness):
    """
    Shuffle the population based on the fitness of the group and numba compatility
    Args:
        x_i: first move
        a_i: left intercept
        d_i: right intercept
        fitness
    Returns:

    """
    indices = np.arange(x_i.shape[0])
    np.random.shuffle(indices)
    x_i = x_i[indices]
    d_i = d_i[indices]
    a_i = a_i[indices]
    fitness = fitness[indices]

    return x_i, a_i, d_i, fitness




def main_loop_iterated(x_i, d_i, a_i, fitnessIN, fitnessOUT, fitnessToT,store_interaction, surplus,\
                       frame_a, frame_x, frame_d,frame_fitnessToT,frame_surplus,\
                        group_size, number_groups, num_interactions, period,mu, step_size,coupled, to_migrate, transfert_multiplier, truc, tracking):


    for i in range(0, period, 1):
        # store the data
        frame_a, frame_x, frame_d, frame_fitnessToT,frame_surplus = store_data(x_i, d_i, a_i, fitnessToT,surplus, frame_a, frame_x, frame_d, \
                   frame_fitnessToT, frame_surplus,i)

        #Migration(Coupled)
        if coupled:
            x_i, d_i, a_i,fitnessToT, fitnessOUT, fitnessIN, surplus = migration(x_i, d_i, a_i, fitnessToT, fitnessOUT, fitnessIN, surplus, number_groups, group_size, to_migrate)


        #Social Dilemma
        x_i, d_i, a_i, store_interaction, fitnessIN,surplus = IN_social_dilemma(x_i, d_i, a_i, store_interaction, fitnessIN, number_groups, group_size,\
                                                                    num_interactions, transfert_multiplier,surplus)

        fitnessToT = fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, number_groups, group_size, truc, num_interactions)

        #Migration (Decoupled)
        if not coupled:
             x_i, d_i, a_i,fitnessToT, fitnessOUT, fitnessIN, surplus = migration(x_i, d_i, a_i, fitnessToT, fitnessOUT, fitnessIN, surplus, number_groups, group_size, to_migrate)

        #Reproduction
        x_i, d_i, a_i = reproduction_pop(x_i, d_i, a_i, fitnessToT,number_groups, mu, step_size)

        tracking[0] = i/period

    return frame_a, frame_x, frame_d


def loop_iterated(group_size, number_groups, num_interactions, period,mu, step_size,coupled, to_migrate, transfert_multiplier, truc,\
                    to_average,tracking, x_i_value, choice):

    for i in range(1, (to_average + 1), 1):
        frame_a, frame_x, frame_d, frame_fitnessToT,frame_surplus, index = create_frames(period,group_size,number_groups)

        x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT,surplus \
            = create_initial_pop(group_size, number_groups, num_interactions, transfert_multiplier, x_i_value, choice)

        frame_a, frame_x, frame_d = main_loop_iterated(x_i, d_i, a_i, fitnessIN, fitnessOUT, fitnessToT, store_interaction,surplus, \
                           frame_a, frame_x, frame_d, frame_fitnessToT,frame_surplus,\
                           group_size, number_groups, num_interactions, period, mu, step_size, coupled, to_migrate,transfert_multiplier, truc,tracking)


        if i == 1:
            frame_x_store = frame_x[index, :]
            frame_a_store = frame_a[index, :]
            frame_d_store = frame_d[index, :]
            frame_surplus_store = frame_surplus[index, :]
        if i > 1:
            frame_x_store = np.hstack((frame_x_store, frame_x[index, :]))
            frame_a_store = np.hstack((frame_a_store, frame_a[index, :]))
            frame_d_store = np.hstack((frame_d_store, frame_d[index, :]))
            frame_surplus_store = np.hstack((frame_surplus_store, frame_surplus[index, :]))
        tracking[1]= i/ to_average

    return frame_x_store, frame_a_store, frame_d_store, frame_surplus_store


def launch_sim_iterated(group_size, number_groups, num_interactions, period, mu, step_size, \
                                coupled, to_migrate, transfert_multiplier, truc,to_average,tracking,x_i_value,choice):


    dir_path = os.path.dirname(os.path.abspath(__file__))

    frame_x_store, frame_a_store, frame_d_store, frame_surplus_store = loop_iterated(\
                    group_size, number_groups, num_interactions, period,mu, step_size,coupled, to_migrate, transfert_multiplier, truc,\
                    to_average,tracking,x_i_value,choice)

    # List of file names to delete
    file_names = ['frame_a.npy', 'frame_x.npy', 'frame_d.npy', 'frame_surplus.npy']

    for file_name in file_names:
        file_path = os.path.join(dir_path, file_name)

        if os.path.exists(file_path):

            os.remove(file_path)

    np.save(os.path.join(dir_path, 'frame_a.npy'), frame_a_store)
    np.save(os.path.join(dir_path, 'frame_x.npy'), frame_x_store)
    np.save(os.path.join(dir_path, 'frame_d.npy'), frame_d_store)
    np.save(os.path.join(dir_path, 'frame_surplus.npy'), frame_surplus_store)


