import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time
from numba.core.registry import CPUDispatcher

def flush(func):
    def flush_numba_cache(func):
        dispatcher = CPUDispatcher(func)
        for sig in list(dispatcher.overloads.keys()):
            del dispatcher.overloads[sig]
        dispatcher.recompile()

    for your_numba_function in [SF.main_loop_group_competition]:
        flush_numba_cache(your_numba_function)

flush(SF.main_loop_group_competition)


group_size = 24
number_groups = 40
num_interactions = 1
to_migrate = 8

period = 35000
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
lambda_param = 10
theta = 0.5

#run the main loop
start = time.time()
SF.main_loop_group_competition(group_size, number_groups, 1, period, frame_a, frame_x, frame_d, frame_t,\
                                frame_u, frame_v, frame_fitnessToT, mu, step_size,\
                                coupled, to_migrate, transfert_multiplier, truc, lambda_param, theta)



end = time.time()

index = np.linspace(0, frame_x.shape[0] - 1, 75).astype(int)
frame_x= frame_x[index, :]
frame_a = frame_a[index, :]
frame_d = frame_d[index, :]
frame_t = frame_t[index, :]
frame_u = frame_u[index, :]
frame_v = frame_v[index, :]
frame_fitnessToT = frame_fitnessToT[index, :]



print("Execution time: ", end - start,"for", period, "iterations.")

start = time.time()
np.save('frame_a.npy', frame_a)
np.save('frame_x.npy', frame_x)
np.save('frame_d.npy', frame_d)


end = time.time()
print("Saving time: ", end - start)


GC.create_frame_x_graph_2()
GC.create_graph_pop_type_2()