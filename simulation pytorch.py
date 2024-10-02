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
    surplus = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
    fitnessIN = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
    fitnessOUT = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
    fitnessToT = torch.zeros((number_groups, group_size), dtype=torch.float32, device=device)
    return x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT,surplus

def create_frames(period, group_size, number_groups, device='cpu'):
    """
    This function creates the tensors to store the result of the simulation.

    Parameters:
    period (int): The number of periods in the simulation.
    group_size (int): The size of each group in the population.
    number_groups (int): The total number of groups in the population.
    device (str): The device to store the tensors ('cpu' or 'mps').
    """
    # Initialize the frames
    frame_a = torch.zeros((period, group_size * number_groups), dtype=torch.float32, device=device)
    frame_x = torch.zeros((period, group_size * number_groups), dtype=torch.float32, device=device)
    frame_d = torch.zeros((period, group_size * number_groups), dtype=torch.float32, device=device)
    frame_surplus = torch.zeros((period, group_size * number_groups), dtype=torch.float32, device=device)
    frame_fitnessToT = torch.zeros((period, group_size * number_groups), dtype=torch.float32, device=device)

    return frame_a, frame_x, frame_d, frame_fitnessToT, frame_surplus

frame_a, frame_x, frame_d, frame_fitnessToT, frame_surplus = create_frames(100000, 24, 40, device='cpu ')
print(frame_a)
print(frame_a.device)

@torch.jit.script
def store_data_torch(frame_a: torch.Tensor,frame_x: torch.Tensor,frame_d: torch.Tensor,
        x_i: torch.Tensor,d_i: torch.Tensor,a_i: torch.Tensor,fitnessToT: torch.Tensor,surplus: torch.Tensor):
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
    frame_a = torch.cat(frame_a, a_i.flatten())
    frame_x = torch.cat(frame_x, x_i.flatten())
    frame_d = torch.cat(frame_d, d_i.flatten())
    frame_fitnessToT = torch.cat(frame_fitnessToT, fitnessToT.flatten())
    frame_surplus = surplus.flatten()
    return frame_a, frame_x, frame_d, frame_fitnessToT, frame_surplus

tensor = torch.cat((tensor, new_tensor))



























