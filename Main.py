import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time

# base parameters

group_size = 24
number_groups = 40
num_interactions = 10
number_of_interaction = num_interactions
to_migrate = 8

period = 100
mu = 0.2
step_size = 0.025
truc = 0.5

frame_a = np.empty((period, (group_size * number_groups)))
frame_x = np.empty((period, (group_size * number_groups)))
frame_d = np.empty((period, (group_size * number_groups)))

#run the main loop
start = time.time()
x_i, d_i, a_i, fitness = SF.main_loop(period, 2, frame_a, frame_x, frame_d, mu, step_size, to_migrate, number_of_interaction, truc, group_size, number_groups, num_interactions)
end = time.time()

#print(fitness)
print("Execution time: ", end - start,"for", period, "iterations.")

start = time.time()

np.save('frame_a.npy', frame_a)
np.save('frame_x.npy', frame_x)
np.save('frame_d.npy', frame_d)

end = time.time()
print("Saving time: ", end - start)

print(frame_a)
GC.create_frame_x_graph_2()
#GC.create_graph_pop_type_2()