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
    #device = torch.device("cpu")
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

def create_storage(period : int, group_size, number_groups, device = 'cpu'):
    """
        This function creates the arrays to store the result of the simulation.

        Parameters:
        period (int): The number of periods in the simulation.
        group_size (int): The size of each group in the population.
        number_groups (int): The total number of groups in the population.
        """
    storage_x = torch.zeros((period, group_size * number_groups), device=device)
    storage_d = torch.zeros((period, group_size * number_groups), device=device)
    storage_a = torch.zeros((period, group_size * number_groups), device=device)
    storage_store_interaction = torch.zeros((period, group_size * number_groups), device=device)
    storage_fitnessIN = torch.zeros((period, group_size * number_groups), device=device)
    storage_fitnessOUT = torch.zeros((period, group_size * number_groups), device=device)
    storage_fitnessToT = torch.zeros((period, group_size * number_groups), device=device)


    return storage_x, storage_d, storage_a, storage_store_interaction, storage_fitnessIN, storage_fitnessOUT, storage_fitnessToT


@torch.jit.script
def store_data(x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT,
                    storage_x, storage_d, storage_a, storage_store_interaction, storage_fitnessIN,
                    storage_fitnessOUT, storage_fitnessToT, period : int):
    """
    Store the data in flattened tensors at the index of the current period.
    Returns:
    Flattened torch.Tensors
    """
    # Store the data in the flattened tensors
    storage_x[period] = x_i.flatten()
    storage_d[period] = d_i.flatten()
    storage_a[period] = a_i.flatten()
    #storage_store_interaction[period] = store_interaction.flatten()
    storage_fitnessIN[period] = fitnessIN.flatten()
    storage_fitnessOUT[period] = fitnessOUT.flatten()
    storage_fitnessToT[period] = fitnessToT.flatten()

    return


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
    number_groups : number of groups (int)
    group_size : size of the groups (int)
    to_migrate : number of players to migrate (int)
    """
    if to_migrate > group_size:
        raise ValueError("The number of migrants is greater than the group size")

    if to_migrate == 0:
        return

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


    # Remplacer les positions des migrants par zéro dans les groupes
    x_i[:, :to_migrate] = 0.0
    d_i[:, :to_migrate] = 0.0
    a_i[:, :to_migrate] = 0.0
    fitnessToT[:, :to_migrate] = 0.0
    fitnessOUT[:, :to_migrate] = 0.0
    fitnessIN[:, :to_migrate] = 0.0


    # Aplatir les migrants et les mélanger
    total_migrants = number_groups * to_migrate
    temp_x_i_flat = temp_x_i.reshape(total_migrants)
    temp_d_i_flat = temp_d_i.reshape(total_migrants)
    temp_a_i_flat = temp_a_i.reshape(total_migrants)
    temp_fitnessToT_flat = temp_fitnessToT.reshape(total_migrants)
    temp_fitnessOUT_flat = temp_fitnessOUT.reshape(total_migrants)
    temp_fitnessIN_flat = temp_fitnessIN.reshape(total_migrants)


    # Mélanger les migrants
    shuffle_indices = torch.randperm(total_migrants, device=device)

    temp_x_i_flat = temp_x_i_flat[shuffle_indices]
    temp_d_i_flat = temp_d_i_flat[shuffle_indices]
    temp_a_i_flat = temp_a_i_flat[shuffle_indices]
    temp_fitnessToT_flat = temp_fitnessToT_flat[shuffle_indices]
    temp_fitnessOUT_flat = temp_fitnessOUT_flat[shuffle_indices]
    temp_fitnessIN_flat = temp_fitnessIN_flat[shuffle_indices]


    # Reshaper les migrants en (number_groups, to_migrate)
    temp_x_i = temp_x_i_flat.reshape(number_groups, to_migrate)
    temp_d_i = temp_d_i_flat.reshape(number_groups, to_migrate)
    temp_a_i = temp_a_i_flat.reshape(number_groups, to_migrate)
    temp_fitnessToT = temp_fitnessToT_flat.reshape(number_groups, to_migrate)
    temp_fitnessOUT = temp_fitnessOUT_flat.reshape(number_groups, to_migrate)
    temp_fitnessIN = temp_fitnessIN_flat.reshape(number_groups, to_migrate)


    # Redistribuer les migrants dans les groupes
    x_i[:, :to_migrate] = temp_x_i
    d_i[:, :to_migrate] = temp_d_i
    a_i[:, :to_migrate] = temp_a_i
    fitnessToT[:, :to_migrate] = temp_fitnessToT
    fitnessOUT[:, :to_migrate] = temp_fitnessOUT
    fitnessIN[:, :to_migrate] = temp_fitnessIN


    return



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

    # Mélange des joueurs au sein de chaque groupe en place
    rand_values = torch.rand(number_groups, group_size, device=device)
    indices = rand_values.argsort(dim=1)
    x_i.copy_(x_i.gather(1, indices))
    d_i.copy_(d_i.gather(1, indices))
    a_i.copy_(a_i.gather(1, indices))









