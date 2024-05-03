import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time

# base parameters

group_size = 24
number_groups = 40
num_interactions = 1
number_of_interaction = num_interactions
to_migrate = 0

period = 50000
mu = 0.02
step_size = 0.025
truc = 0.5

frame_a = np.zeros((960, period))
frame_x = np.zeros((960, period))
frame_d = np.zeros((960, period))

#run the main loop
start = time.time()
x_i, d_i, a_i, fitness = SF.main_loop(period, 2, frame_a, frame_x, frame_d, mu, step_size, to_migrate, number_of_interaction, truc, group_size, number_groups, num_interactions)
end = time.time()

print(fitness)
print("Execution time: ", end - start,"for", period, "iterations.")

GC.create_frame_x_graph_2()
GC.create_graph_pop_type_2()

start = time.time()

np.save('frame_a.npy', frame_a)
np.save('frame_x.npy', frame_x)
np.save('frame_d.npy', frame_d)

end = time.time()
print("Saving time: ", end - start)

