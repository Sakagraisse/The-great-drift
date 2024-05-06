from random import random as rand, shuffle, randint, choice
import numpy as np
import pandas as pd
import numba as nb
import time
from tqdm import tqdm

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
    x_i = np.ones((number_groups, group_size), dtype=np.float64)
    d_i = np.ones((number_groups, group_size), dtype=np.float64)
    a_i = np.zeros((number_groups, group_size), dtype=np.float64)
    store_interaction = np.empty((number_groups, group_size,num_interactions), dtype=np.float64)
    endo_fit = np.empty((number_groups, group_size,num_interactions), dtype=np.float64)
    fitness = np.empty((number_groups, group_size), dtype=np.float64)

    return x_i, d_i, a_i, store_interaction, endo_fit, fitness


@nb.jit(nopython=True)
def mutate(value, mu, step_size):
    """
    Applies a mutation to the given value based on the mutation probability mu.
    """
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
        if value == 0 and np.random.random() < mu/2:
            new_value = value + step_size
        elif value == 1 and np.random.random() < mu/2:
            new_value = value - step_size
        else:
            # The value remains the same
            new_value = value

    return new_value


@nb.jit(nopython=True)
def setdiff1d_numba(arr1, arr2):
    return np.asarray(list(set(arr1) - set(arr2)))
@nb.jit(nopython=True)
def meta_pop():
    """
    Return a vector of groups to form the meta-population for out-group interactions
    """
    #generate a random number between 0, 20 , 40
    to_move = np.random.choice(np.array([0, 20, 40]))
    #generate a vector of to_move elements with random ints between 0 and 40
    indices = np.random.choice(np.arange(40), to_move, replace=False)
    # Get the remaining indices
    remain_indices = setdiff1d_numba(np.arange(40), indices)

    return indices, remain_indices

@nb.jit(nopython=True)
def out_group_competition(t_i,u_i,v_j,fitness2,group_size, number_groups,transfert_multiplier):
    """
    Create the out-group competition
    """
    indices,remain = meta_pop()
    for j in range(len(indices),2):
        indices = np.arange(group_size)
        np.random.shuffle(indices)
        t_i[j,:] = t_i[j,indices]
        u_i[j,:] = u_i[j,indices]
        v_j[j,:] = v_j[j,indices]
        fitness2[j,:] = np.empty(group_size, dtype=np.float64)
        np.random.shuffle(indices)
        t_i[(j+1),:] = t_i(j+1,indices)
        u_i[(j+1),:] = u_i(j+1,indices)
        v_j[(j+1),:] = v_j(j+1,indices)
        fitness2[(j+1),:] = np.empty(group_size, dtype=np.float64)
        for i in range(group_size):
            temp_x_i = t_i[j,i]
            temp_y_i = u_i[(j+1),i] + (v_j[(j+1),i] - u_i[(j+1),i]) * temp_x_i
            fitness2[j,i] = 1 - temp_x_i + temp_y_i*transfert_multiplier
            fitness2[(j+1),i] = 1 - temp_y_i[(j+1),i] + temp_x_i*transfert_multiplier

    return fitness2




@nb.jit(nopython=True)
def migration(x_i, d_i, a_i, store_interaction, endo_fit, fitness, to_migrate,number_groups, group_size):
    """
    Create the migration of pop groups
    """

    temp_x_i = np.empty((number_groups,to_migrate), dtype=np.float64)
    temp_d_i = np.empty((number_groups,to_migrate), dtype=np.float64)
    temp_a_i = np.empty((number_groups,to_migrate), dtype=np.float64)
    temp_fitness = np.empty((number_groups,to_migrate), dtype=np.float64)

    # extract from the pop the migrtantes
    for i in range(number_groups):
        indices = np.arange(group_size)
        np.random.shuffle(indices)
        x_i[i,:] = x_i[i,indices]
        d_i[i,:] = d_i[i,indices]
        a_i[i,:] = a_i[i,indices]
        fitness[i,:] = fitness[i,indices]
        temp_x_i[i,:] = x_i[i,0:to_migrate]
        x_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)
        temp_d_i[i,:] = d_i[i,0:to_migrate]
        d_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)
        temp_a_i[i,:] = a_i[i,0:to_migrate]
        a_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)
        temp_fitness[i,:] = fitness[i,0:to_migrate]
        fitness[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)


    temp_x_i = temp_x_i.flatten()
    temp_d_i = temp_d_i.flatten()
    temp_a_i = temp_a_i.flatten()
    temp_fitness = temp_fitness.flatten()
    indices2 = np.arange((to_migrate*number_groups))
    np.random.shuffle(indices2)
    temp_x_i = temp_x_i[indices2]
    temp_d_i = temp_d_i[indices2]
    temp_a_i = temp_a_i[indices2]
    temp_fitness = temp_fitness[indices2]



    for j in range(number_groups):
        x_i[j,0:to_migrate] = temp_x_i[j*to_migrate:(j+1)*to_migrate]
        d_i[j,0:to_migrate] = temp_d_i[j*to_migrate:(j+1)*to_migrate]
        a_i[j,0:to_migrate] = temp_a_i[j*to_migrate:(j+1)*to_migrate]
        fitness[j,0:to_migrate] = temp_fitness[j*to_migrate:(j+1)*to_migrate]

    return x_i, d_i, a_i, fitness


@nb.jit(nopython=True)
def social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitness, number_groups, group_size, num_interactions, transfert_multiplier, truc):
    """
    Create the social dilemma
    """
    for j in range(number_groups):
        indices = np.arange(group_size)
        np.random.shuffle(indices)
        x_i[j,:] = x_i[j,indices]
        d_i[j,:] = d_i[j,indices]
        a_i[j,:] = a_i[j,indices]
        fitness[j,:] = np.empty(group_size, dtype=np.float64)
        store_interaction[j,:,:] = np.empty((group_size,num_interactions), dtype=np.float64)
        endo_fit[j,:,:] = np.empty((group_size,num_interactions), dtype=np.float64)


        for i in range(0, group_size, 2):
            store_interaction[j, i, 0] = x_i[j, i]
            store_interaction[j, (i + 1), 0] = a_i[j, (i + 1)] + (d_i[j, (i + 1)] - a_i[j, (i + 1)]) * \
                                               store_interaction[j, i, 0]
            endo_fit[j, i, 0] = 1 - store_interaction[j, i, 0] + store_interaction[j, (i + 1), 0] * transfert_multiplier
            endo_fit[j, (i + 1), 0] = 1 - store_interaction[j, (i + 1), 0] + store_interaction[j, i, 0] * transfert_multiplier
            fitness[j, i] = 1 - store_interaction[j, i, 0] + store_interaction[j, (i + 1), 0] * transfert_multiplier
            fitness[j, (i + 1)] = 1 - store_interaction[j, (i + 1), 0] + store_interaction[j, i, 0] * transfert_multiplier

            if num_interactions > 1:
                for k in range(1,num_interactions):
                    store_interaction[j, i, k] = a_i[j, i] + (d_i[j, i] - a_i[j, i]) * store_interaction[j, (i+1), k-1]
                    store_interaction[j, (i+1), k] = a_i[j, (i+1)] + (d_i[j, (i+1)] - a_i[j, (i+1)]) * store_interaction[j, i, k]

                    endo_fit[j, i, k] = 1 - store_interaction[j, i, k] + store_interaction[j, (i+1), k] * transfert_multiplier
                    endo_fit[j, (i+1), k] = 1 - store_interaction[j, (i+1), k] + store_interaction[j, i, k] * transfert_multiplier
                    fitness[j, i] +=  1 - store_interaction[j, i, k] + store_interaction[j, (i+1), k] * transfert_multiplier
                    fitness[j, (i+1)] += 1 - store_interaction[j, (i+1), k] + store_interaction[j, i, k] * transfert_multiplier
            fitness[j, i] = (1 - truc) * num_interactions + truc * fitness[j, i]
            fitness[j, (i + 1)] = (1 - truc) * num_interactions + truc * fitness[j, (i + 1)]



    return x_i, d_i, a_i, store_interaction, endo_fit, fitness



@nb.jit(nopython=True)
def replace_group_comp(x_i, d_i, a_i, fitness, mu, step_size,competition_list, remaining,victory):

    """
    Replace the group
    """
    for j in range(len(remaining), 2):
        if victory[j]:
            won = int(competition_list[j])
            lost = int(competition_list[j + 1])
        else:
            won = int(competition_list[j + 1])
            lost = int(competition_list[j])
        x_i[lost], d_i[lost], a_i[lost] = reproduction_one_group(x_i[won], d_i[won], a_i[won], fitness[won], mu,
                                                                 step_size)

    for j in remaining:
        x_i[j], d_i[j], a_i[j] = reproduction_one_group(x_i[j], d_i[j], a_i[j], fitness[j], mu, step_size)

    return x_i, d_i, a_i, fitness


@nb.jit(nopython=True)
def reproduction(x_i, d_i, a_i, fitness, mu, step_size):
    """
    Reproduction of the population
    """
    for j in range(x_i.shape[0]):
        x_i[j], d_i[j], a_i[j] = reproduction_one_group(x_i[j], d_i[j], a_i[j], fitness[j], mu, step_size)

    return x_i, d_i, a_i

@nb.jit(nopython=True)
def reproduction_one_group(v1,v2,v3, fitness, mu, step_size):
    """
    Reproduction of one group of the population
    """
    #for x_i
    v1 = return_pop_vector_Ui(v1, fitness)
    #for d_i
    v2 = return_pop_vector_Ui(v2, fitness)
    #for a_i
    v3 = return_pop_vector_Ui(v3, fitness)
    for i in range(v1.shape[0]):
        v1[i] = mutate(v1[i], mu, step_size)
        v2[i] = mutate(v2[i], mu, step_size)
        v3[i] = mutate(v3[i], mu, step_size)
    return v1, v2, v3


@nb.jit(nopython=True)
def main_loop(period, transfert_multiplier, frame_a, frame_x, frame_d, mu, step_size, to_migrate, truc, group_size, number_groups, num_interactions):
    # create the initial population
    x_i, d_i, a_i, store_interaction, endo_fit, fitness = create_initial_pop(group_size, number_groups, num_interactions)

    #main loop
    for i in range(0, period, 1):
        print("Progress:", round((i / period)*100) ,"%")

        # store the data
        store_data(x_i, d_i, a_i, frame_a, frame_x, frame_d, i)

        # migration
        x_i, d_i, a_i,fitness = migration(x_i, d_i, a_i, store_interaction, endo_fit, fitness, to_migrate, number_groups, group_size)
        # social dilemna
        x_i, d_i, a_i, store_interaction, endo_fit, fitness = social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitness, number_groups, group_size, num_interactions, transfert_multiplier, truc)
        # reproduction
        x_i, d_i, a_i = reproduction(x_i, d_i, a_i, fitness, mu, step_size)
        #print(fitness)

    return x_i, d_i, a_i, fitness

@nb.jit(nopython=True)
def store_data(x_i, d_i, a_i, frame_a, frame_x, frame_d, period):
    """
    Store the data in a numpy array
    """
    frame_a[period, :] = a_i.flatten()
    frame_x[period, :] = x_i.flatten()
    frame_d[period, :] = d_i.flatten()

    return frame_a, frame_x, frame_d

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

    sum = np.sum(fitness)
    prob = fitness / sum
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



@nb.jit(nopython=True)
def group_comp(fitnessToT,competition_list,remaining,group_size, truc, transfert_multiplier, num_interactions,theta, victory):


    delta = group_size * truc*(num_interactions*(transfert_multiplier-1)+1(transfert_multiplier+1))
    for i in range(len(competition_list),2):
        fitnessTot_G1 = np.sum(fitnessToT[competition_list[i]])
        fitnessTot_G2 = np.sum(fitnessToT[competition_list[i+1]])
        proba_competition =( np.abs(fitnessTot_G1 - fitnessTot_G2) / delta)/(theta + np.abs(fitnessTot_G1 - fitnessTot_G2) / delta)
        if np.random.random() < proba_competition:
            proba_victory = 1 / (1 + np.exp((theta / delta) * (fitnessTot_G1 - fitnessTot_G2)))
            if np.random.random() < proba_victory:
                victory[i] = True
                victory[i+1] = False
            else:
                victory[i] = False
                victory[i+1] = True

        else:
            pass
    return victory
