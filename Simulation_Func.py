
import numpy as np

import numba as nb


################
# Create base player
################

#Create numba arrays
#seed np.random
np.random.seed(0)

#POP creation

@nb.jit(nopython=True)
def create_initial_pop(group_size, number_groups, num_interactions):
    """
    Creates the initial population of players.
    """
    x_i = np.ones((number_groups, group_size), dtype=np.float64)
    d_i = np.ones((number_groups, group_size), dtype=np.float64)
    a_i = np.zeros((number_groups, group_size), dtype=np.float64)
    t_i = np.ones((number_groups, group_size), dtype=np.float64)
    u_i = np.zeros((number_groups, group_size), dtype=np.float64)
    v_j = np.ones((number_groups, group_size), dtype=np.float64)

    store_interaction = np.empty((number_groups, group_size,num_interactions), dtype=np.float64)
    endo_fit = np.empty((number_groups, group_size,num_interactions), dtype=np.float64)
    fitnessIN = np.zeros((number_groups, group_size), dtype=np.float64)
    fitnessOUT = np.zeros((number_groups, group_size), dtype=np.float64)
    fitnessToT = np.zeros((number_groups, group_size), dtype=np.float64)

    return x_i, d_i, a_i, t_i, u_i, v_j, store_interaction, endo_fit, fitnessIN, fitnessOUT, fitnessToT


@nb.jit(nopython=True)
def store_data(x_i, d_i, a_i, t_i, u_i , v_i,fitnessToT , frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, frame_fitnessToT, period):
    """
    Store the data in a numpy array
    """
    frame_a[period, :] = a_i.flatten()
    frame_x[period, :] = x_i.flatten()
    frame_d[period, :] = d_i.flatten()
    frame_t[period, :] = t_i.flatten()
    frame_u[period, :] = u_i.flatten()
    frame_v[period, :] = v_i.flatten()
    frame_fitnessToT[period, :] = fitnessToT.flatten()
    return

@nb.jit(nopython=True)
def setdiff1d_numba(arr1, arr2):
    return np.asarray(list(set(arr1) - set(arr2)))

@nb.jit(nopython=True)
def meta_pop(number_groups):
    """
    Return a vector of groups to form the meta-population for out-group interactions
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
    return indices

@nb.jit(nopython=True)
def move_groups(x_i, d_i, a_i, t_i, u_i, v_i, indices):
    """
    Move groups from one population to another
    """
    x_i = x_i[indices, :]
    d_i = d_i[indices, :]
    a_i = a_i[indices, :]
    t_i = t_i[indices, :]
    u_i = u_i[indices, :]
    v_i = v_i[indices, :]
    return



@nb.jit(nopython=True)
def migration(x_i, d_i, a_i,t_i,u_i,v_i, to_migrate,number_groups, group_size):
    """
    Create the migration of pop groups
    """

    temp_x_i = np.empty((number_groups,to_migrate), dtype=np.float64)
    temp_d_i = np.empty((number_groups,to_migrate), dtype=np.float64)
    temp_a_i = np.empty((number_groups,to_migrate), dtype=np.float64)
    temp_t_i = np.empty((number_groups,to_migrate), dtype=np.float64)
    temp_u_i = np.empty((number_groups,to_migrate), dtype=np.float64)
    temp_v_i = np.empty((number_groups,to_migrate), dtype=np.float64)

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
        x_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)
        temp_d_i[i,:] = d_i[i,0:to_migrate]
        d_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)
        temp_a_i[i,:] = a_i[i,0:to_migrate]
        a_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)
        temp_t_i[i,:] = t_i[i,0:to_migrate]
        t_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)
        temp_u_i[i,:] = u_i[i,0:to_migrate]
        u_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)
        temp_v_i[i,:] = v_i[i,0:to_migrate]
        v_i[i,0:to_migrate] = np.empty(to_migrate, dtype=np.float64)


    temp_x_i = temp_x_i.flatten()
    temp_d_i = temp_d_i.flatten()
    temp_a_i = temp_a_i.flatten()
    temp_t_i = temp_t_i.flatten()
    temp_u_i = temp_u_i.flatten()
    temp_v_i = temp_v_i.flatten()

    indices2 = np.arange((to_migrate*number_groups))
    np.random.shuffle(indices2)
    temp_x_i = temp_x_i[indices2]
    temp_d_i = temp_d_i[indices2]
    temp_a_i = temp_a_i[indices2]
    temp_t_i = temp_t_i[indices2]
    temp_u_i = temp_u_i[indices2]
    temp_v_i = temp_v_i[indices2]

    for j in range(number_groups):
        x_i[j,0:to_migrate] = temp_x_i[j*to_migrate:(j+1)*to_migrate]
        d_i[j,0:to_migrate] = temp_d_i[j*to_migrate:(j+1)*to_migrate]
        a_i[j,0:to_migrate] = temp_a_i[j*to_migrate:(j+1)*to_migrate]
        t_i[j,0:to_migrate] = temp_t_i[j*to_migrate:(j+1)*to_migrate]
        u_i[j,0:to_migrate] = temp_u_i[j*to_migrate:(j+1)*to_migrate]
        v_i[j,0:to_migrate] = temp_v_i[j*to_migrate:(j+1)*to_migrate]
    return x_i, d_i, a_i, t_i, u_i, v_i




@nb.jit(nopython=True)
def IN_social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitnessIN, number_groups, group_size, num_interactions, transfert_multiplier):
    """
    Create the social dilemma
    """
    for j in range(number_groups):
        indices = np.arange(group_size)
        np.random.shuffle(indices)

        fitnessIN[j,:] = np.zeros(group_size, dtype=np.float64)
        store_interaction[j,:,:] = np.zeros((group_size,num_interactions), dtype=np.float64)

        for i in range(0, group_size, 2):
            p1 = indices[i]
            p2 = indices[i+1]
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

    return


@nb.jit(nopython=True)
def OUT_social_dilemma(t_i,u_i,v_j,fitnessOUT,group_size, number_groups,transfert_multiplier,indices_group):
    """
    Create the out-group competition
    """
    for j in range(0,number_groups,2):
        indices_in_group = np.arange(group_size)
        np.random.shuffle(indices_in_group)
        g1 = indices_group[j]
        g2 = indices_group[j+1]
        for i in range(0,group_size):
            p = indices_in_group[i]
            temp_x = t_i[g1,p]
            temp_y = u_i[g2,p] + (v_j[g2,p] - u_i[g2,p]) * temp_x
            fitnessOUT[g1,p] = 1 - temp_x + temp_y * transfert_multiplier
            fitnessOUT[g1,p] = 1 - temp_y + temp_x * transfert_multiplier

    return

@nb.jit(nopython=True)
def fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, number_groups, group_size,truc,num_interactions,compi = 0):
    """
    Calculate the total fitness
    """
    for j in range(number_groups):
        for i in range(group_size):
            fitnessToT[j,i] = (1-truc)*(num_interactions + compi) + truc *(fitnessIN[j,i] + truc*fitnessOUT[j,i])
    return

@nb.jit(nopython=True)
def intergroup_comp(fitnessToT, number_groups, group_size, truc, transfert_multiplier, num_interactions,theta, victory,indices_group,lambda_param,do_compete):

    delta = group_size * truc*(num_interactions*(transfert_multiplier-1)+(transfert_multiplier+1))

    for i in range(0,number_groups,2):
        G1 = indices_group[i]
        G2 = indices_group[i+1]
        fitnessTot_G1 = np.sum(fitnessToT[G1])
        fitnessTot_G2 = np.sum(fitnessToT[G2])
        proba_competition = (np.abs(fitnessTot_G2 - fitnessTot_G1) / delta)/(theta + np.abs(fitnessTot_G2 - fitnessTot_G1) / delta)
        if np.random.random() < proba_competition:
            do_compete[G1] = True
            do_compete[G2] = True
            proba_victory = 1 / (1 + np.exp((lambda_param / delta) * (fitnessTot_G2 - fitnessTot_G1)))
            if np.random.random() < proba_victory:
                victory[G1] = True
                victory[G2] = False
            else:
                victory[G1] = False
                victory[G2] = True

        else:
            do_compete[G1] = False
            do_compete[G2] = False
    return

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
def replace_group_comp(x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, indices_group, victory, do_compete, mu, step_size,number_groups):

    """
    Replace the group
    """
    for j in range(0,number_groups, 2):
        G1 = indices_group[j]
        G2 = indices_group[j+1]
        if do_compete[G1] and do_compete[G2]:
            if victory[G1]:
                won = G1
                lost = G2
            elif victory[G2]:
                won = G2
                lost = G1
            else:
                raise ValueError("compete but no winner")
            x_i[lost,:], d_i[lost,:], a_i[lost,:] = reproduction_one_group(x_i[won,:], d_i[won,:], a_i[won,:], fitnessToT[won,:], mu, step_size)
            #same with the rest of parameters
            t_i[lost,:], u_i[lost,:], v_i[lost,:] = reproduction_one_group(t_i[won,:], u_i[won,:], v_i[won,:], fitnessToT[won,:], mu, step_size)
        elif not do_compete[G1] and not do_compete[G2]:
            x_i[G1,:], d_i[G1,:], a_i[G1,:] = reproduction_one_group(x_i[G1,:], d_i[G1,:], a_i[G1,:], fitnessToT[G1,:], mu, step_size)
            t_i[G1,:], u_i[G1,:], v_i[G1,:] = reproduction_one_group(t_i[G1,:], u_i[G1,:], v_i[G1,:], fitnessToT[G1,:], mu, step_size)
            x_i[G2,:], d_i[G2,:], a_i[G2,:] = reproduction_one_group(x_i[G2,:], d_i[G2,:], a_i[G2,:], fitnessToT[G2,:], mu, step_size)
            t_i[G2,:], u_i[G2,:], v_i[G2,:] = reproduction_one_group(t_i[G2,:], u_i[G2,:], v_i[G2,:], fitnessToT[G2,:], mu, step_size)

        else :
            # raise an error in due form
            raise ValueError("The groups are not in the same state")
    return


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
def main_loop_group_competition(group_size, number_groups, num_interactions, period, frame_a, frame_x, frame_d, frame_t,\
                                frame_u, frame_v, frame_fitnessToT, mu, step_size,\
                                coupled, to_migrate, transfert_multiplier, truc, lambda_param, theta):

    x_i, d_i, a_i, t_i, u_i, v_i, store_interaction, endo_fit, fitnessIN, fitnessOUT, fitnessToT \
        = create_initial_pop(group_size, number_groups, num_interactions)

    victory = np.zeros(number_groups, dtype=np.bool_)
    do_compete = np.zeros(number_groups, dtype=np.bool_)

    for i in range(0, period, 1):
        print("Progress:", round((i / period) * 100), "%")
        # store the data
        store_data(x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, \
                   frame_fitnessToT, i)

        indices = meta_pop(number_groups)
        move_groups(x_i, d_i, a_i, t_i, u_i, v_i, indices)

        if coupled:
            migration(x_i, d_i, a_i, t_i, u_i, v_i, to_migrate, number_groups, group_size)

        IN_social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitnessIN, number_groups, group_size,
                          num_interactions, transfert_multiplier)

        indices_group = np.arange(number_groups)
        np.random.shuffle(indices_group)

        OUT_social_dilemma(t_i, u_i, v_i, fitnessOUT, group_size, number_groups, transfert_multiplier, indices_group)

        if not coupled:
            migration(x_i, d_i, a_i, t_i, u_i, v_i, to_migrate, number_groups, group_size)

        fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, number_groups, group_size, truc, num_interactions,1)

        intergroup_comp(fitnessToT, number_groups, group_size, truc, transfert_multiplier, num_interactions, theta,
                        victory, indices_group, lambda_param, do_compete)

        replace_group_comp(x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, indices_group, victory, do_compete, mu, step_size,number_groups)

    return


@nb.jit(nopython=True)
def main_loop_iterated(group_size, number_groups, num_interactions, period, frame_a, frame_x, frame_d, frame_t, \
                                frame_u, frame_v, frame_fitnessToT, mu, step_size, \
                                coupled, to_migrate, transfert_multiplier, truc):

    x_i, d_i, a_i, t_i, u_i, v_i, store_interaction, endo_fit, fitnessIN, fitnessOUT, fitnessToT \
        = create_initial_pop(group_size, number_groups, num_interactions)

    do_compete = np.zeros(number_groups, dtype=np.bool_)
    victory = np.zeros(number_groups, dtype=np.bool_)

    for i in range(0, period, 1):
        print("Progress:", round((i / period) * 100), "%")
        # store the data
        store_data(x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, \
                   frame_fitnessToT, i)


        #Movement in metapopulation
        #indices = meta_pop(number_groups)
        #move_groups(x_i, d_i, a_i, t_i, u_i, v_i, indices)

        #Migration(Coupled)
        if coupled:
            x_i, d_i, a_i, t_i, u_i, v_i = migration(x_i, d_i, a_i, t_i, u_i, v_i, to_migrate, number_groups, group_size)

        #Social Dilemma
        IN_social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitnessIN, number_groups, group_size,\
                                                                    num_interactions, transfert_multiplier)

        #Migration (Decoupled)
        if not coupled:
             migration(x_i, d_i, a_i, t_i, u_i, v_i, to_migrate, number_groups, group_size)

        #Total fitness Calulation per individual
        fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, number_groups, group_size, truc, num_interactions)


        #Reproduction
        indices_group = np.arange(number_groups)
        np.random.shuffle(indices_group)
        replace_group_comp(x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, indices_group, victory, do_compete, mu, step_size, number_groups)


    return

@nb.jit(nopython=True)
def joint_scenario(group_size, number_groups, num_interactions, period, frame_a, frame_x, frame_d,
                   frame_t, frame_u, frame_v, frame_fitnessToT, mu, step_size, \
                   coupled, to_migrate, transfert_multiplier, truc, lambda_param, theta):

    x_i, d_i, a_i, t_i, u_i, v_i, store_interaction, endo_fit, fitnessIN, fitnessOUT, fitnessToT \
        = create_initial_pop(group_size, number_groups, num_interactions)

    x_i = np.zeros((number_groups, group_size), dtype=np.float64)
    d_i = np.zeros((number_groups, group_size), dtype=np.float64)
    a_i = np.zeros((number_groups, group_size), dtype=np.float64)

    victory = np.zeros(number_groups, dtype=np.bool_)
    do_compete = np.zeros(number_groups, dtype=np.bool_)

    for i in range(0, period, 1):
        print("Progress:", round((i / period) * 100), "%")
        # store the data
        store_data(x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, frame_a, frame_x, frame_d, frame_t, frame_u, frame_v, \
                   frame_fitnessToT, i)

        indices = meta_pop(number_groups)
        move_groups(x_i, d_i, a_i, t_i, u_i, v_i, indices)

        if coupled:
            migration(x_i, d_i, a_i, t_i, u_i, v_i, to_migrate, number_groups, group_size)

        IN_social_dilemma(x_i, d_i, a_i, store_interaction, endo_fit, fitnessIN, number_groups, group_size,
                          num_interactions, transfert_multiplier)

        indices_group = np.arange(number_groups)
        np.random.shuffle(indices_group)

        OUT_social_dilemma(t_i, u_i, v_i, fitnessOUT, group_size, number_groups, transfert_multiplier,
                           indices_group)

        if not coupled:
            migration(x_i, d_i, a_i, t_i, u_i, v_i, to_migrate, number_groups, group_size)

        fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, number_groups, group_size, truc, num_interactions,
                               1)

        intergroup_comp(fitnessToT, number_groups, group_size, truc, transfert_multiplier, num_interactions, theta,
                        victory, indices_group, lambda_param, do_compete)

        replace_group_comp(x_i, d_i, a_i, t_i, u_i, v_i, fitnessToT, indices_group, victory, do_compete, mu,
                           step_size, number_groups)

    return


