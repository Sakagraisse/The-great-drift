from Main import *

pop = create_initial_pop()
#print(pop[1][1].x_i)
pop = migration(pop)


pop = social_dilemna(pop, transfert_multiplier)

pop = reproduction(pop)

