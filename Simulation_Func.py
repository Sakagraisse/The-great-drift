import os
import numpy as np
import numba as nb
from math import sqrt
import torch
from typing import Tuple


def activate_torch_mps():
    """
    Activate the MPS backend if available.
    """
    # Check if the MPS backend is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Le backend MPS est disponible. Exécution sur le GPU.")
    else:
        device = torch.device("cpu")
        print("Le backend MPS n'est pas disponible. Exécution sur le CPU.")
    device = torch.device("cpu")
    return device









def create_initial_pop(group_size, number_groups, num_interactions,transfert_multiplier,x_i_value,choice,device = 'cpu'):
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
        x_i = torch.ones(number_groups, group_size, dtype=torch.float32, device=device) * x_i_value
        a_i = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
        d_i = torch.ones((number_groups, group_size), dtype=torch.float32, device=device)

    elif choice == 1:
        x_i = torch.ones(number_groups, group_size, dtype=torch.float32, device=device) * x_i_value
        a_i = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
        d_i = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)

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

        x_i = torch.ones(number_groups, group_size, dtype=torch.float32, device=device) * x_i_value
        a_i = torch.ones((number_groups, group_size), dtype=torch.float32, device=device) * a_hat
        d_i = torch.ones((number_groups, group_size), dtype=torch.float32, device=device)
    else:
        raise ValueError("The choice must be 0, 1 or 2")


    # Initialize the store_interaction, fitnessIN, fitnessOUT, fitnessToT, and surplus arrays
    store_interaction = torch.zeros((number_groups, group_size, num_interactions), dtype=torch.float32, device=device)
    fitnessIN = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
    fitnessOUT = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
    fitnessToT = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
    return x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT

def create_storage(period, group_size, number_groups):
    """
        This function creates the arrays to store the result of the simulation.

        Parameters:
        period (int): The number of periods in the simulation.
        group_size (int): The size of each group in the population.
        number_groups (int): The total number of groups in the population.
        """
    # Initialize the frames
    storage_x = np.zeros((period, (group_size * number_groups)))
    storage_d = np.zeros((period, (group_size * number_groups)))
    storage_a = np.zeros((period, (group_size * number_groups)))
    storage_store_interaction = np.zeros((period, (group_size * number_groups)))
    storage_fitnessIN = np.zeros((period, (group_size * number_groups)))
    storage_fitnessOUT = np.zeros((period, (group_size * number_groups)))
    storage_fitnessToT = np.zeros((period, (group_size * number_groups)))

    return storage_x, storage_d, storage_a, storage_store_interaction, storage_fitnessIN, storage_fitnessOUT, storage_fitnessToT



@torch.jit.script
def store_data_torch(x_i, d_i, a_i, fitnessToT, surplus, fitnessOUT, store_interaction,
    """
    Store the data in flattened tensors.

    Parameters:
    x_i : first move (torch.Tensor)
    a_i : left intercept (torch.Tensor)
    d_i : right intercept (torch.Tensor)
    fitnessToT : total fitness (torch.Tensor)
    surplus : surplus (torch.Tensor)

    Returns:
    Flattened torch.Tensors
    """
    frame_a = a_i.flatten()
    frame_x = x_i.flatten()
    frame_d = d_i.flatten()
    frame_fitnessToT = fitnessToT.flatten()
    frame_surplus = surplus.flatten()
    return frame_a, frame_x, frame_d, frame_fitnessToT, frame_surplus



#@nb.jit(nopython=True)
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

    frame_a_torch, frame_x_torch, frame_d_torch, frame_fitnessToT_torch, frame_surplus_torch = store_data_torch(
        x_i, d_i, a_i, fitnessToT, surplus
    )

    # Convertir les tenseurs en tableaux NumPy
    frame_a_np = frame_a_torch.cpu().numpy()
    frame_x_np = frame_x_torch.cpu().numpy()
    frame_d_np = frame_d_torch.cpu().numpy()
    frame_fitnessToT_np = frame_fitnessToT_torch.cpu().numpy()
    frame_surplus_np = frame_surplus_torch.cpu().numpy()

    # Stocker les données dans les tableaux NumPy de stockage
    frame_a[period, :] = frame_a_np
    frame_x[period, :] = frame_x_np
    frame_d[period, :] = frame_d_np
    frame_fitnessToT[period, :] = frame_fitnessToT_np
    frame_surplus[period, :] = frame_surplus_np
    return frame_a, frame_x, frame_d, frame_fitnessToT,frame_surplus





@torch.jit.script
def setdiff1d_numba(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    """
    Compute the set difference of two tensors using PyTorch.
    Returns the elements that are in arr1 but not in arr2.

    Parameters:
    arr1: torch.Tensor
    arr2: torch.Tensor

    Returns:
    torch.Tensor
    """
    mask = ~torch.isin(arr1, arr2)
    return arr1[mask]




@torch.jit.script
def migration(
    x_i: torch.Tensor,
    d_i: torch.Tensor,
    a_i: torch.Tensor,
    fitnessToT: torch.Tensor,
    fitnessOUT: torch.Tensor,
    fitnessIN: torch.Tensor,
    number_groups: int,
    group_size: int,
    to_migrate: int
):
    """
    Migrate the players between groups using PyTorch tensors.

    Parameters:
    x_i : first move (torch.Tensor)
    d_i : right intercept (torch.Tensor)
    a_i : left intercept (torch.Tensor)
    fitnessToT : total fitness (torch.Tensor)
    fitnessOUT : fitness out group interaction (torch.Tensor)
    fitnessIN : fitness in group interaction (torch.Tensor)
    surplus : surplus of the players (torch.Tensor)
    number_groups : number of groups (int)
    group_size : size of the groups (int)
    to_migrate : number of players to migrate (int)
    """
    if to_migrate > group_size:
        raise ValueError("The number of migrants is greater than the group size")

    if to_migrate == 0:
        return x_i, d_i, a_i, fitnessToT, fitnessOUT, fitnessIN

    device = x_i.device

    # Générer des permutations aléatoires pour chaque groupe
    rand_values = torch.rand(number_groups, group_size, device=device)
    indices = rand_values.argsort(dim=1)

    # Réorganiser les tenseurs en utilisant les indices mélangés
    x_i = x_i.gather(1, indices)
    d_i = d_i.gather(1, indices)
    a_i = a_i.gather(1, indices)
    fitnessToT = fitnessToT.gather(1, indices)
    fitnessOUT = fitnessOUT.gather(1, indices)
    fitnessIN = fitnessIN.gather(1, indices)


    # Extraire les migrants
    temp_x_i = x_i[:, :to_migrate].clone()
    temp_d_i = d_i[:, :to_migrate].clone()
    temp_a_i = a_i[:, :to_migrate].clone()
    temp_fitnessToT = fitnessToT[:, :to_migrate].clone()
    temp_fitnessOUT = fitnessOUT[:, :to_migrate].clone()
    temp_fitnessIN = fitnessIN[:, :to_migrate].clone()
    temp_surplus = surplus[:, :to_migrate].clone()

    # Remplacer les positions des migrants par zéro dans les groupes
    x_i[:, :to_migrate] = 0.0
    d_i[:, :to_migrate] = 0.0
    a_i[:, :to_migrate] = 0.0
    fitnessToT[:, :to_migrate] = 0.0
    fitnessOUT[:, :to_migrate] = 0.0
    fitnessIN[:, :to_migrate] = 0.0
    surplus[:, :to_migrate] = 0.0

    # Aplatir les migrants et les mélanger
    total_migrants = number_groups * to_migrate
    temp_x_i_flat = temp_x_i.reshape(total_migrants)
    temp_d_i_flat = temp_d_i.reshape(total_migrants)
    temp_a_i_flat = temp_a_i.reshape(total_migrants)
    temp_fitnessToT_flat = temp_fitnessToT.reshape(total_migrants)
    temp_fitnessOUT_flat = temp_fitnessOUT.reshape(total_migrants)
    temp_fitnessIN_flat = temp_fitnessIN.reshape(total_migrants)
    temp_surplus_flat = temp_surplus.reshape(total_migrants)

    # Mélanger les migrants
    shuffle_indices = torch.randperm(total_migrants, device=device)

    temp_x_i_flat = temp_x_i_flat[shuffle_indices]
    temp_d_i_flat = temp_d_i_flat[shuffle_indices]
    temp_a_i_flat = temp_a_i_flat[shuffle_indices]
    temp_fitnessToT_flat = temp_fitnessToT_flat[shuffle_indices]
    temp_fitnessOUT_flat = temp_fitnessOUT_flat[shuffle_indices]
    temp_fitnessIN_flat = temp_fitnessIN_flat[shuffle_indices]
    temp_surplus_flat = temp_surplus_flat[shuffle_indices]

    # Reshaper les migrants en (number_groups, to_migrate)
    temp_x_i = temp_x_i_flat.reshape(number_groups, to_migrate)
    temp_d_i = temp_d_i_flat.reshape(number_groups, to_migrate)
    temp_a_i = temp_a_i_flat.reshape(number_groups, to_migrate)
    temp_fitnessToT = temp_fitnessToT_flat.reshape(number_groups, to_migrate)
    temp_fitnessOUT = temp_fitnessOUT_flat.reshape(number_groups, to_migrate)
    temp_fitnessIN = temp_fitnessIN_flat.reshape(number_groups, to_migrate)
    temp_surplus = temp_surplus_flat.reshape(number_groups, to_migrate)

    # Redistribuer les migrants dans les groupes
    x_i[:, :to_migrate] = temp_x_i
    d_i[:, :to_migrate] = temp_d_i
    a_i[:, :to_migrate] = temp_a_i
    fitnessToT[:, :to_migrate] = temp_fitnessToT
    fitnessOUT[:, :to_migrate] = temp_fitnessOUT
    fitnessIN[:, :to_migrate] = temp_fitnessIN
    surplus[:, :to_migrate] = temp_surplus

    return x_i, d_i, a_i, fitnessToT, fitnessOUT, fitnessIN







@torch.jit.script
def IN_social_dilemma(
    x_i: torch.Tensor,
    d_i: torch.Tensor,
    a_i: torch.Tensor,
    store_interaction: torch.Tensor,
    fitnessIN: torch.Tensor,
    number_groups: int,
    group_size: int,
    num_interactions: int,
    transfert_multiplier: float,
    surplus: torch.Tensor
    ):
    """
    Perform the in-group social dilemma using PyTorch tensors.

    Parameters:
    x_i : first move (torch.Tensor)
    d_i : right intercept (torch.Tensor)
    a_i : left intercept (torch.Tensor)
    store_interaction : store the interaction (torch.Tensor)
    fitnessIN : fitness in group interaction (torch.Tensor)
    number_groups : number of groups (int)
    group_size : size of the groups (int)
    num_interactions : number of interactions (int)
    transfert_multiplier : transfer multiplier (float)
    surplus : surplus of the players (torch.Tensor)

    Returns:
    Updated tensors: x_i, d_i, a_i, store_interaction, fitnessIN, surplus
    """
    device = x_i.device

    # Mélange des joueurs au sein de chaque groupe
    rand_values = torch.rand(number_groups, group_size, device=device)
    indices = rand_values.argsort(dim=1)
    x_i = x_i.gather(1, indices)
    d_i = d_i.gather(1, indices)
    a_i = a_i.gather(1, indices)

    # Génération des indices des joueurs appariés
    num_pairs = group_size // 2
    p1_indices = torch.arange(0, num_pairs * 2, 2, device=device)
    p2_indices = torch.arange(1, num_pairs * 2, 2, device=device)

    # Extraction des variables pour p1 et p2
    x_i_p1 = x_i[:, p1_indices]
    x_i_p2 = x_i[:, p2_indices]
    a_i_p1 = a_i[:, p1_indices]
    a_i_p2 = a_i[:, p2_indices]
    d_i_p1 = d_i[:, p1_indices]
    d_i_p2 = d_i[:, p2_indices]

    # Initialisation de store_interaction pour p1 et p2
    store_interaction_p1 = torch.zeros(number_groups, num_pairs, num_interactions, device=device)
    store_interaction_p2 = torch.zeros(number_groups, num_pairs, num_interactions, device=device)

    store_interaction_p1[:, :, 0] = x_i_p1
    store_interaction_p2[:, :, 0] = a_i_p2 + (d_i_p2 - a_i_p2) * store_interaction_p1[:, :, 0]

    # Initialisation de fitnessIN
    fitnessIN_p1 = 1 - store_interaction_p1[:, :, 0] + store_interaction_p2[:, :, 0] * transfert_multiplier
    fitnessIN_p2 = 1 - store_interaction_p2[:, :, 0] + store_interaction_p1[:, :, 0] * transfert_multiplier

    # Interactions supplémentaires si num_interactions > 1
    if num_interactions > 1:
        for k in range(1, num_interactions):
            store_interaction_p1[:, :, k] = a_i_p1 + (d_i_p1 - a_i_p1) * store_interaction_p2[:, :, k - 1]
            store_interaction_p2[:, :, k] = a_i_p2 + (d_i_p2 - a_i_p2) * store_interaction_p1[:, :, k]

            fitnessIN_p1 += 1 - store_interaction_p1[:, :, k] + store_interaction_p2[:, :, k] * transfert_multiplier
            fitnessIN_p2 += 1 - store_interaction_p2[:, :, k] + store_interaction_p1[:, :, k] * transfert_multiplier

    # Calcul du surplus
    surplus_p1 = (torch.sum(store_interaction_p1, dim=2) * transfert_multiplier) / num_interactions
    surplus_p2 = (torch.sum(store_interaction_p2, dim=2) * transfert_multiplier) / num_interactions

    # Mise à jour de fitnessIN et surplus
    fitnessIN.index_copy_(1, p1_indices, fitnessIN_p1)
    fitnessIN.index_copy_(1, p2_indices, fitnessIN_p2)
    surplus.index_copy_(1, p1_indices, surplus_p1)
    surplus.index_copy_(1, p2_indices, surplus_p2)

    # Mise à jour de store_interaction
    store_interaction.index_copy_(1, p1_indices, store_interaction_p1)
    store_interaction.index_copy_(1, p2_indices, store_interaction_p2)

    # Gestion des joueurs non appariés si group_size est impair
    if group_size % 2 != 0:
        last_player_index = group_size - 1
        fitnessIN[:, last_player_index] = 1.0  # Valeur par défaut
        surplus[:, last_player_index] = 0.0
        store_interaction[:, last_player_index, :] = 0.0

    return x_i, d_i, a_i, store_interaction, fitnessIN, surplus



import torch

@torch.jit.script
def fitnessToT_calculation(
    fitnessIN: torch.Tensor,
    fitnessOUT: torch.Tensor,
    fitnessToT: torch.Tensor,
    truc: float,
    num_interactions: int
) -> torch.Tensor:
    """
    Calculate the total fitness using PyTorch tensors.

    Parameters:
    fitnessIN : fitness in group interaction (torch.Tensor)
    fitnessOUT : fitness out group interaction (torch.Tensor)
    fitnessToT : total fitness (torch.Tensor)
    truc : parameter (float)
    num_interactions : number of interactions (int)

    Returns:
    Updated fitnessToT tensor.
    """
    fitnessToT = (1 - truc) * num_interactions + truc * (fitnessIN + fitnessOUT)
    return fitnessToT



@torch.jit.script
def mutate(value: torch.Tensor, mu: float, step_size: float) -> torch.Tensor:
    """
    Applies a mutation to the given tensor of values based on the mutation probability mu.

    Parameters:
    value : The tensor of values to mutate (torch.Tensor)
    mu : The mutation probability (float)
    step_size : The step size of the mutation (float)

    Returns:
    A tensor with mutated values.
    """
    # Vérification des valeurs d'entrée
    if torch.any(value < 0) or torch.any(value > 1):
        raise ValueError("All values must be within the range [0, 1].")
    if mu < 0 or mu > 1:
        raise ValueError("The mutation probability mu must be within the range [0, 1].")
    if step_size <= 0 or step_size > 1:
        raise ValueError("The step size must be within the range (0, 1].")

    # Création d'un tenseur pour les nouvelles valeurs
    new_value = value.clone()

    # Masques pour les différentes conditions
    mask_mid = (value > 0) & (value < 1)  # Valeurs strictement entre 0 et 1
    mask_zero = (value == 0)              # Valeurs égales à 0
    mask_one = (value == 1)               # Valeurs égales à 1

    # Génération de nombres aléatoires pour la décision de mutation
    rand_uniform = torch.rand_like(value)

    # Mutation pour les valeurs entre 0 et 1
    mutate_mask_mid = (rand_uniform < mu) & mask_mid

    # Décider la direction du pas (-step_size ou +step_size)
    random_directions = torch.randint(0, 2, value.shape, device=value.device)
    step_direction = step_size * (2.0 * random_directions.float() - 1.0)

    # Appliquer la mutation aux valeurs qui mutent
    new_value[mutate_mask_mid] = value[mutate_mask_mid] + step_direction[mutate_mask_mid]

    # Mutation pour les valeurs à la frontière (0 ou 1)
    mutate_mask_zero = (rand_uniform < mu / 2) & mask_zero
    mutate_mask_one = (rand_uniform < mu / 2) & mask_one

    # Appliquer la mutation pour les valeurs égales à 0
    new_value[mutate_mask_zero] = value[mutate_mask_zero] + step_size

    # Appliquer la mutation pour les valeurs égales à 1
    new_value[mutate_mask_one] = value[mutate_mask_one] - step_size

    # S'assurer que les valeurs restent dans [0, 1]
    new_value = torch.clamp(new_value, 0.0, 1.0)

    return new_value

@torch.jit.script
def return_pop_vector_Ui(
    value: torch.Tensor,
    fitness: torch.Tensor
) -> torch.Tensor:
    """
    Return a vector of the population based on the fitness of the group.

    Parameters:
    value : torch.Tensor
        Tensor of parameter values.
    fitness : torch.Tensor
        Fitness values corresponding to each individual in the group.

    Returns:
    torch.Tensor
        A new tensor where each element is selected based on the fitness probabilities.
    """
    # Calcul des probabilités de sélection
    total_group_fitness = fitness.sum()
    if total_group_fitness == 0.0:
        # Si le total de fitness est zéro, utiliser des probabilités uniformes
        prob = torch.ones_like(fitness) / fitness.size(0)
    else:
        prob = fitness / total_group_fitness

    # Taille du groupe
    test_size = value.size(0)

    # Échantillonnage des indices en fonction des probabilités
    selected_indices = torch.multinomial(prob, test_size, replacement=True)

    # Sélection des valeurs correspondantes
    test = value[selected_indices]

    return test

@torch.jit.script
def reproduction_one_group(
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    fitnessToT: torch.Tensor,
    mu: float,
    step_size: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reproduction of one group of the population.

    Parameters:
    v1 : torch.Tensor
        First vector of parameter.
    v2 : torch.Tensor
        Second vector of parameter.
    v3 : torch.Tensor
        Third vector of parameter.
    fitnessToT : torch.Tensor
        Total fitness.
    mu : float
        Mutation probability.
    step_size : float
        Step size of the mutation.

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Updated vectors v1, v2, v3.
    """
    # Sélection des individus en fonction du fitness
    v1 = return_pop_vector_Ui(v1, fitnessToT)
    v2 = return_pop_vector_Ui(v2, fitnessToT)
    v3 = return_pop_vector_Ui(v3, fitnessToT)

    # Application de la mutation en utilisant la fonction 'mutate' vectorisée
    v1 = mutate(v1, mu, step_size)
    v2 = mutate(v2, mu, step_size)
    v3 = mutate(v3, mu, step_size)

    return v1, v2, v3

@torch.jit.script
def custom_random_choice(prob: torch.Tensor) -> int:
    """
    Custom random choice function that selects an index based on the given probabilities.

    Parameters:
    prob : Probability tensor (torch.Tensor)

    Returns:
    index : Selected index (int)
    """
    rand = torch.rand(1).item()
    cum_prob = torch.cumsum(prob, dim=0)
    for i in range(cum_prob.size(0)):
        if rand < cum_prob[i].item():
            return i
    return prob.size(0) - 1




@torch.jit.script
def costum_shuffle_pop(
    x_i: torch.Tensor,
    a_i: torch.Tensor,
    d_i: torch.Tensor,
    fitness: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shuffle the population based on the fitness of the group.

    Parameters:
    x_i : torch.Tensor
        First move.
    a_i : torch.Tensor
        Left intercept.
    d_i : torch.Tensor
        Right intercept.
    fitness : torch.Tensor
        Fitness values.

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Shuffled tensors x_i, a_i, d_i, fitness.
    """
    indices = torch.randperm(x_i.size(0))
    x_i = x_i[indices]
    a_i = a_i[indices]
    d_i = d_i[indices]
    fitness = fitness[indices]

    return x_i, a_i, d_i, fitness


@torch.jit.script
def reproduction_pop(
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    fitnessToT: torch.Tensor,
    number_groups: int,
    mu: float,
    step_size: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes three vectors and reproduces them based on the fitness of the group

    Parameters:
    v1 : first vector of parameter (torch.Tensor)
    v2 : second vector of parameter (torch.Tensor)
    v3 : third vector of parameter (torch.Tensor)
    fitnessToT : total fitness (torch.Tensor)
    number_groups : number of groups (int)
    mu : mutation probability (float)
    step_size : step size of the mutation (float)
    """
    for j in range(number_groups):
        v1_group, v2_group, v3_group = reproduction_one_group(
            v1[j, :],
            v2[j, :],
            v3[j, :],
            fitnessToT[j, :],
            mu,
            step_size
        )
        v1[j, :] = v1_group
        v2[j, :] = v2_group
        v3[j, :] = v3_group
    return v1, v2, v3











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

        fitnessToT = fitnessToT_calculation(fitnessIN, fitnessOUT, fitnessToT, truc, num_interactions)


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

group_size = 24
number_groups = 40
num_interactions = 100
period = 1000
mu = 0.02
step_size = 0.025
coupled = True
to_migrate = 8
transfert_multiplier = 2
truc = 0.5
to_average = 1
tracking = np.zeros(2)
x_i_value = 1
choice = 0
activate_torch_mps()
#timing the simulation
import time
start = time.time()
x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT,surplus \
            = create_initial_pop(group_size, number_groups, num_interactions, transfert_multiplier, x_i_value, choice)
frame_a, frame_x, frame_d, frame_fitnessToT,frame_surplus, index = create_frames(period,group_size,number_groups)

frame_a, frame_x, frame_d = main_loop_iterated(x_i, d_i, a_i, fitnessIN, fitnessOUT, fitnessToT,store_interaction, surplus,\
                       frame_a, frame_x, frame_d,frame_fitnessToT,frame_surplus,\
                        group_size, number_groups, num_interactions, period,mu, step_size,coupled, to_migrate, transfert_multiplier, truc, tracking)

end = time.time()
print("Time taken: ", end - start)
