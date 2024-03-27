################
# import libraries and dependencies
################

from collections import Counter
from random import random, shuffle, seed

from numpy.random import normal


################
# Create base parameters
################
number_of_indiv = 100
transfert_multiplier = 1.2
period = 100


################
# Create base player
################

class Player:
    ID = 0

    def __init__(self, type=None, x_i=0.7, y_i=0.8):
        self.type = type
        self.id = Player.ID
        Player.ID += 1
        self.fitness = 1
        self.x_i = x_i
        self.y_i = y_i
        self.age = 0


################
#create the initial population
################
# function that create the starting population.
# There is a number of individual set by : number_of_indiv
def create_initial_pop(number_of_indiv):
    # Tous les joueurs sont créés en tant que "hawk"
    return [Player("hawk") for _ in range(number_of_indiv)]

################
#Social dilemna
################
# function that simulate the social dilemna
# take pop as input and return the population with updated fitness
# take individuals by pair in the order of the list
#they play a sequential game where the first player tranfert x_i to the second player
# the amount is multiplied by transfert_multiplier
# the second player transfert back y_i to the first player
# the amount is multiplied by transfert_multiplier
# the fitness of the players is updated with the amount they have

def social_dilemna(pop, transfert_multiplier):
    for i in range(0, len(pop), 2):
        x_i = pop[i].x_i
        y_i = pop[i].y_i
        pop[i].fitness +=  - x_i + y_i * transfert_multiplier
        pop[i + 1].fitness += - y_i + x_i * transfert_multiplier
    return pop

################
# update age
################
# function that update the age of the population
# take pop as input and return the population with updated age
def update_age(pop):
    for i in pop:
        i.age += 1
    return pop

################
# Main loop
################

# function that simulate the game
def main_loop(period, number_of_indiv, transfert_multiplier):
    #main loop
    for i in range(period):
        # create the initial population
        pop = create_initial_pop(number_of_indiv)
        # shuffle the population
        shuffle(pop)
        # calculate fitness
        pop = social_dilemna(pop, transfert_multiplier)
        # update age
        pop = update_age(pop)
    return pop


################
# Main loop
################
#Run the main loop
to_display = main_loop(period, number_of_indiv, transfert_multiplier)
#display the result
print(to_display[0].fitness)





