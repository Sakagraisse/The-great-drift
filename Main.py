################
# import libraries and dependencies
################

from collections import Counter
from random import random as rand, shuffle, randint, choice
import numpy as np
import pandas as pd
import time

################
# Create base player
################

class Player:
    ID = 0

    def __init__(self, x_i=1, d_i=1, a_i=0, num_interactions=10):

        self.id = Player.ID
        Player.ID += 1
        self.fitness = 1
        self.x_i = x_i
        self.a_i = a_i
        self.d_i = d_i
        # create array name store_interaction of length num_interactions
        self.store_interaction = [0] * num_interactions
        self.endo_fit = [0] * num_interactions



def create_initial_pop():
    """
    Creates the initial population of players.
    """
    pop = [[Player() for _ in range(24)] for _ in range(40)]
    return pop


def mutate(value,mu, step_size):
    """
    Applies a mutation to the given value based on the mutation probability mu.
    """
    if value not in {0, 1}:
        if rand() < mu:
            # Mutation: decide the step direction (up or down)
            step_direction = choice([-step_size, step_size])
            # Apply the mutation while staying within the [0,1] boundaries
            new_value = min(1, max(0, value + step_direction))
        else:
            # No mutation, the value remains the same
            new_value = value
    else:
        # The value is at the boundary of the grid, it can only move in one direction
        if value == 0 and rand() < mu/2:
            new_value = value + step_size
        elif value == 1 and rand() < mu/2:
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

def migration(pop,to_migrate=8):
    """
    Migrate 8 to 16 member of each group to another group
    """
    migrants = []
    pop2 = []
    to_add = [0] * 40
    for j in range(0,40,1):
        #choose the number of migration
        temp = to_migrate
        temps_pop = pop[j]
        to_add[j] = temp
        # shuffle the population
        shuffle(temps_pop)
        # take the first temp individuals
        temp_migrants = temps_pop[:temp]
        # remove the first temp individuals
        temps_pop = temps_pop[temp:]
        # add to migrants by concatening
        migrants = migrants + temp_migrants
        # add the new population to the new list
        pop2.append(temps_pop)

    # shuffle the migrants
    shuffle(migrants)
    # raise error if the number of migrant is not equal to the sum of the to_add list
    if len(migrants) != len(migrants):
        raise ValueError("The number of migrants is not equal to the sum of the to_add list")
    # add the migrants to the new population
    for j in range(0,40,1):
        numb = to_add[j]
        pop2[j] = migrants[:numb] + pop2[j]
        migrants = migrants[numb:]
    # raise error if the numbner of pop is not equal to 24 for each of the 40 value in pop2
    for i in range(0,40,1):
        if len(pop2[i]) != 24:
            raise ValueError("The number of pop is not equal to 24 for each of the 40 value in pop2")

    return pop2


def social_dilemna(pop, transfert_multiplier=2, number_of_interaction=10,truc=0.9):
    """
    Social dilemna
    """

    for j in range(0, 40, 1):
        temp_pop = pop[j]
        for i in range(0, 24, 2):
            # select the first player
            rd_value = randint(0, 1)
            p1 = i + rd_value
            p2 = i + 1 - rd_value
            for k in range(0, number_of_interaction, 1):
                if i == 0:
                    temp_pop[p1].store_interaction[k] = temp_pop[p1].x_i
                    temp_pop[p2].store_interaction[k] = temp_pop[p2].a_i + (temp_pop[p2].d_i - temp_pop[p2].a_i) * temp_pop[p1].x_i

                else:
                    temp_pop[p1].store_interaction[k] = temp_pop[p1].a_i + (temp_pop[p1].d_i - temp_pop[p2].a_i) * temp_pop[p2].store_interaction[k-1]
                    temp_pop[p2].store_interaction[k] = temp_pop[p2].a_i + (temp_pop[p2].d_i - temp_pop[p2].a_i) * temp_pop[p1].store_interaction[k]
                temp_pop[p1].endo_fit[k] = 1 - temp_pop[p1].store_interaction[k] + temp_pop[p2].store_interaction[k] * transfert_multiplier
                temp_pop[p2].endo_fit[k] = 1 - temp_pop[p2].store_interaction[k] + temp_pop[p1].store_interaction[k] * transfert_multiplier

            temp_pop[p1].fitness = (1-truc)*number_of_interaction + truc * sum(temp_pop[p1].endo_fit)
            temp_pop[p2].fitness = (1-truc)*number_of_interaction + truc * sum(temp_pop[p2].endo_fit )
    pop[j] = temp_pop

    return pop

def reproduction(pop,mu, step_size):
    """
    Reproduction of the population
    """
    for j in range(0,40,1):
        sum_fitness = sum(player.fitness for player in pop[j])
        for i in range(0,24,1):
            prob = pop[j][i].fitness / sum_fitness

            #for x_i
            if rand() < prob:
                x_i = mutate(pop[j][i].x_i, mu, step_size)
            else:
                x_i = round(rand() * 20) / 20


            #for d_i
            if rand() < prob:
                d_i = mutate(pop[j][i].d_i, mu, step_size)
            else:
                d_i = round(rand() * 20) / 20

            #for a_i
            if rand() < prob:
                a_i = mutate(pop[j][i].a_i, mu, step_size)
            else:
                a_i = round(rand() * 20) / 20

            pop[j][i] = Player(x_i=x_i, d_i=d_i, a_i=a_i)

    return pop
            





################
# Main loop
################

# function that simulate the game
def main_loop(period, transfert_multiplier, frame_a, frame_x, frame_d, mu, step_size,to_migrate,number_of_interaction,truc):
    # create the initial population
    pop = create_initial_pop()
    #meta pop
    #pop = meta_pop(pop)

    #migration
    pop = migration(pop,to_migrate)

    #social dilemna
    pop = social_dilemna(pop, transfert_multiplier, number_of_interaction, truc)

    store_data(pop, frame_a, frame_x, frame_d, 0)
    #reproduction
    pop = reproduction(pop, mu, step_size)


    #main loop
    for i in range(1,period,1):
        #print(i)
        # shuffle the population
        #shuffle(pop)
        #pop = meta_pop(pop)
        # migration
        pop = migration(pop,to_migrate)
        # social dilemna
        pop = social_dilemna(pop, transfert_multiplier, number_of_interaction, truc)
        # store the data
        store_data(pop, frame_a, frame_x, frame_d, i)
        # reproduction
        pop = reproduction(pop, mu, step_size)


    return pop


def store_data(pop,frame_a,frame_x,frame_d,period):
    """
    Store the data in a dataframe file
    """
    counter = 0
    #iterate on ever pop number
    for j in range(0,40,1):
        #iterate on every player in the population
        for i in range(0,24,1):
            #store the data in the dataframe
            frame_a.iloc[(counter-1), period] = pop[j][i].a_i
            frame_x.iloc[(counter-1), period] = pop[j][i].x_i
            frame_d.iloc[(counter-1), period] = pop[j][i].d_i
            counter += 1



################
# Main loop
################

transfert_multiplier = 2
period = 500
to_migrate = 0
number_of_interaction = 10
mu = 0.2
step_size = 0.025
# weithing parameter for the fitness
truc = 0.5
# create a dataframe of 960 rows and period columns
frame_a = pd.DataFrame(np.zeros((960, period)))
frame_x = pd.DataFrame(np.zeros((960, period)))
frame_d = pd.DataFrame(np.zeros((960, period)))

start = time.time()
pop = main_loop(period, transfert_multiplier, frame_a, frame_x, frame_d,mu, step_size,to_migrate,number_of_interaction,truc)

end = time.time()
print("Time taken: ", end - start, "for", period, "iterations.")

start = time.time()
#store frame_a, frame_x, frame_d in a csv file unsing df.to_pickle('file_name.csv')
frame_a.to_csv('frame_a.csv', index=False)
frame_x.to_csv('frame_x.csv', index=False)
frame_d.to_csv('frame_d.csv', index=False)

end = time.time()
print("Time taken to store the data: ", end - start, "for", period, "iterations.")




