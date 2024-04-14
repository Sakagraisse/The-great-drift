################
# import libraries and dependencies
################

from collections import Counter
from random import random, shuffle, randint




################
# Create base player
################

class Player:
    ID = 0

    def __init__(self, x_i=0.7, d_i=0.9, a_i=0.6, num_interactions=100):

        self.id = Player.ID
        Player.ID += 1
        self.fitness = 1
        self.x_i = x_i
        self.a_i = a_i
        self.d_i = d_i
        # create array name store_interaction of length num_interactions
        self.store_interaction = [0] * num_interactions



def create_initial_pop():
    """
    Creates the initial population of players.
    """
    pop = [[Player() for _ in range(24)] for _ in range(40)]
    return pop


def mutate(value):
    """
    Applies a mutation to the given value based on the mutation probability mu.
    """
    if value not in {0, 1}:
        if random.random() < mu:
            # Mutation: decide the step direction (up or down)
            step_direction = random.choice([-step_size, step_size])
            # Apply the mutation while staying within the [0,1] boundaries
            new_value = min(1, max(0, value + step_direction))
        else:
            # No mutation, the value remains the same
            new_value = value
    else:
        # The value is at the boundary of the grid, it can only move in one direction
        if value == 0 and random.random() < mu/2:
            new_value = value + step_size
        elif value == 1 and random.random() < mu/2:
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

def migration(pop):
    """
    Migrate 8 to 16 member of each group to another group
    """
    migrants = []
    pop2 = []
    to_add = [0] * 40
    for j in range(0,40,1):
        temp = randint(8,16)
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
        print(len(pop2[i]))
        if len(pop2[i]) != 24:
            raise ValueError("The number of pop is not equal to 24 for each of the 40 value in pop2")

    return pop2


def social_dilemna(pop, transfert_multiplier, number_of_interaction=1):
    """
    Social dilemna
    """

    for j in range(0, 40, 1):
        temp_pop = pop[j]
        for i in range(0, 24, 2):
            for k in range(0, number_of_interaction, 1):
                # select the first player
                rd = randint(0, 1)
                p1 = i+rd
                p2 = i+1-rd
                if i == 0:
                    temp_pop[p1].store_interaction[i] = temp_pop[p1].x_i
                    temp_pop[p2].store_interaction[i] = temp_pop[p2].a_i + (temp_pop[p2].d_i - temp_pop[p2].a_i) * temp_pop[p1].x_i
                else:
                    temp_pop[p1].store_interaction[i] = temp_pop[p1].a_i + (temp_pop[p1].d_i - temp_pop[p2].a_1) * temp_pop[p2].store_interaction[i-1]
                    temp_pop[p2].store_interaction[i] = temp_pop[p2].a_i + (temp_pop[p2].d_i - temp_pop[p2].a_i) * temp_pop[p1].store_interaction[i]
                endo_fit_1 = 1 - temp_pop[p1].store_interaction[i] + temp_pop[p2].store_interaction[i] * transfert_multiplier
                endo_fit_2 = 1 - temp_pop[p2].store_interaction[i] + temp_pop[p1].store_interaction[i] * transfert_multiplier
                temp_pop[p1].fitness = (1-truc)*number_of_interaction + truc * sum(endo_fit_1)
                temp_pop[p2].fitness = (1-truc)*number_of_interaction + truc * sum(endo_fit_2)
        pop[j] = temp_pop

    return pop

def reproduction(pop):
    """
    Reproduction of the population
    """
    for j in range(0,40,1):
        sum_fitness = sum(player.fitness for player in pop[j])
        for i in range(0,24,1):
            prob = pop[j][i].fitness / sum_fitness

            #for x_i
            if randint(0,1) < prob:
                pop[j][i] = Player(x_i = pop[j][i].x_i)
            else:
                pop[j][i] = Player()

            #for d_i
            if randint(0,1) < prob:
                pop[j][i] = Player(d_i = pop[j][i].d_i)
            else:
                pop[j][i] = Player()

            #for a_i
            if randint(0,1) < prob:
                pop[j][i] = Player(a_i = pop[j][i].a_i)
            else:
                pop[j][i] = Player()
            





################
# Main loop
################

# function that simulate the game
def main_loop(period, transfert_multiplier):
    # create the initial population
    pop = create_initial_pop()
    #meta pop
    pop = meta_pop(pop)
    #migration
    pop = migration(pop)
    #social dilemna
    pop = social_dilemna(pop, transfert_multiplier)
    #reproduction
    pop = reproduction(pop)

    #main loop
    for i in range(period):
        # create the initial population
        pop = create_initial_pop()
        # shuffle the population
        shuffle(pop)
        # calculate fitness
        pop = social_dilemna(pop, transfert_multiplier)
        # update age

    return pop


################
# Main loop
################

transfert_multiplier = 2
period = 100
dim = 2
number_of_interaction = 100
mu = 0.02
step_size = 0.025
# weithing parameter for the fitness
truc = 0.9

#pop = main_loop(100, 2)
#print(pop)





