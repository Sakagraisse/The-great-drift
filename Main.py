import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time

# base parameters

group_size = 24
number_groups = 40
num_interactions = 10
to_migrate = 16

period = 100
mu = 0.02
step_size = 0.025
truc = 0.5

frame_a = np.empty((period, (group_size * number_groups)))
frame_x = np.zeros((period, (group_size * number_groups)))
frame_d = np.empty((period, (group_size * number_groups)))
frame_t = np.empty((period, (group_size * number_groups)))
frame_u = np.empty((period, (group_size * number_groups)))
frame_v = np.empty((period, (group_size * number_groups)))
frame_fitnessToT = np.empty((period, (group_size * number_groups)))

coupled = True
transfert_multiplier = 2

#run the main loop
start = time.time()
SF.main_loop_iterated(group_size, number_groups, num_interactions, period, frame_a, frame_x, frame_d, frame_t, \
                                frame_u, frame_v, frame_fitnessToT, mu, step_size, \
                                coupled, to_migrate, transfert_multiplier, truc)


print(frame_x[1, :])
end = time.time()




print("Execution time: ", end - start,"for", period, "iterations.")

start = time.time()
np.save('frame_a.npy', frame_a)
np.save('frame_x.npy', frame_x)
np.save('frame_d.npy', frame_d)


end = time.time()
print("Saving time: ", end - start)


GC.create_frame_x_graph_2()
GC.create_graph_pop_type_2()