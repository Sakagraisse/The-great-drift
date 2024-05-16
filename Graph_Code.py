import sys
import os


import numpy as np

import matplotlib.pyplot as plt
import numba as nb


#@nb.jit(nopython=True)
def create_frame_x_graph_2(period):
    # Import frame_x from csv file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_x = np.load(os.path.join(dir_path, 'frame_x.npy'))

    frame_x_bins = np.zeros((75, 10))
    bins = np.arange(0, 1.1, 0.1)
    for i in range(0, 75, 1):
        hist, bin_edges = np.histogram(frame_x[i, :], bins)
        frame_x_bins[i, :] = hist

    data = frame_x_bins
    #transposing the data
    data = data.T
    plt.figure(figsize=(12, 8))
    plt.imshow(data, cmap='Greys', aspect='auto')
    plt.colorbar(label='Value')
    plt.xticks(range(data.shape[1]), rotation=90)
    # Calculate the indices for start, quarter, three quarters and end
    indices = [0, period // 3, 2 * period // 4, (period - 1)]

    # Select the corresponding periods
    selected_periods = np.arange(period)
    selected_periods = selected_periods[indices]
    # Set the x-ticks at the selected indices
    plt.xticks(indices, selected_periods)
    custom_ticks = np.arange(0, 1.1, 0.1)  # replace with your desired ticks
    custom_ticks = np.round(custom_ticks, 1)  # round to one decimal place
    plt.yticks(np.linspace(0, data.shape[0] - 1, len(custom_ticks)), custom_ticks)
    plt.gca().invert_yaxis()  # Invert y-axis
    plt.xlabel('Period')
    plt.ylabel('Bins')
    plt.title('Graph')
    #plt.show()
    #save the graph
    dir_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dir_path, 'frame_x.png'),dpi=75)

    return

@nb.jit(nopython=True)
def function_1(a,b):
    y = a/(a+1-b)
    return y
def create_graph_pop_type_2():
    # Importer frame_a et frame_d à partir de fichiers csv
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_a = np.load(os.path.join(dir_path, 'frame_a.npy'))
    frame_d = np.load(os.path.join(dir_path, 'frame_d.npy'))

    # Créer un tableau de 75 colonnes et 6 lignes
    frame_a_bins = np.zeros((75, 9))  # Increase the size of the array to accommodate the new categories
    for j in range(0, 75, 1):
        for i in range(0, len(frame_a[j]), 1):
            if frame_a[j, i] == 0 and frame_d[j, i] == 0:
                frame_a_bins[j, 0] += 1
            elif frame_a[j, i] == 0 and frame_d[j, i] < 1 and frame_d[j, i] > 0:
                frame_a_bins[j, 1] += 1
            elif frame_a[j, i] > 0 and frame_a[j, i] < 1 \
                    and frame_d[j, i] > 0 and frame_d[j, i] < 1 \
                    and function_1(frame_a[j, i], frame_d[j, i]) <= 0.1:
                frame_a_bins[j, 2] += 1
            elif frame_a[j, i] > 0 and frame_a[j, i] < 1 \
                    and frame_d[j, i] > 0 and frame_d[j, i] < 1 \
                    and function_1(frame_a[j, i], frame_d[j, i]) < 0.9\
                    and function_1(frame_a[j, i], frame_d[j, i]) > 0.1\
                    and frame_a[j, i] <= frame_d[j, i]:
                frame_a_bins[j, 3] += 1
            elif frame_a[j, i] == 0 and frame_d[j, i] == 1:
                frame_a_bins[j, 4] += 1
            elif frame_a[j, i] < 1 and frame_a[j, i] > 0 \
                    and frame_d[j, i] < 1 and frame_d[j, i] > 0 \
                    and function_1(frame_a[j, i], frame_d[j, i]) >= 0.9 :
                frame_a_bins[j, 5] += 1
            elif frame_a[j, i] > 0 and frame_a[j, i] < 1  and frame_d[j, i] == 1:
                frame_a_bins[j, 6] += 1
            elif frame_a[j, i] == 1 and frame_d[j, i] == 1:
                frame_a_bins[j, 7] += 1
            else:
                frame_a_bins[j, 8] += 1







    for i in range(0, 75, 1):
        sum_q = np.sum(frame_a_bins[i, :])
        frame_a_bins[i, :] = frame_a_bins[i, :] / sum_q
    data = frame_a_bins
    #transposing the data
    data = data.T
    colors = ['red', 'lightgreen', 'purple', 'blue', 'darkgreen', 'lightblue', 'orange', 'brown', 'grey']
    # Nombre de groupes (c'est-à-dire nombre de barres empilées)
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(data.shape[1]), data[0, :], color=colors[0])
    bottom = data[0, :]
    for i in range(1, data.shape[0]):
        plt.bar(np.arange(data.shape[1]), data[i, :], bottom=bottom, color=colors[i])
        bottom += data[i, :]
    plt.title('Stacked Bar Graph')
    plt.xlabel("Generation")
    plt.ylabel("Proportion")
    plt.legend(["Unconditionally selfish", "De−escalators",  "Quasi-de-escalator", "Ambiguous", "Perfect reciprocators", "Quasi-de-escalator","Escalators",
                "Unconditionally generous", "Other"], loc="upper left",
               bbox_to_anchor=(1, 1))

    #plt.show()
    # Sauvegarder le graphique
    dir_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dir_path, 'frame_a.png'), dpi=75)

    return
