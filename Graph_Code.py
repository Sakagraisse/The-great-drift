import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import numba as nb

#@nb.jit(nopython=True)
def create_frame_x_graph_2():
    # Import frame_x from csv file
    frame_x = np.load('frame_x.npy')

    # Division of columns into 75 groups
    index = np.linspace(0, frame_x.shape[0]-1, 75).astype(int)
    print(index)
    # Create an empty dataframe for frame_x_shorten
    frame_x_shorten = frame_x[index,:]
    frame_x_bins = np.zeros((75, 10))
    bins = np.arange(0, 1.1, 0.1)
    for i in range(0, 75, 1):
        hist, bin_edges = np.histogram(frame_x_shorten[i, :], bins)
        frame_x_bins[i, :] = hist
    print("lol")
    data = frame_x_bins

    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='Greys', aspect='auto')
    plt.colorbar(label='Value')
    plt.xticks(range(data.shape[1]), rotation=0)
    plt.yticks(range(data.shape[0]), bins[:-1])
    plt.gca().invert_yaxis()  # Invert y-axis
    plt.xlabel('Period')
    plt.ylabel('Bins')
    plt.title('Graph')
    plt.show()
    #save the graph
    plt.savefig('frame_x.png')

    return

@nb.jit(nopython=True)
def function_1(a,d,x):
    y = a + (d-a)*x
    return y
def create_graph_pop_type_2():
    # Importer frame_a et frame_d à partir de fichiers csv
    frame_a = np.load('frame_a.npy')
    frame_d = np.load('frame_d.npy')

    # Créer des groupes de 75 colonnes comme dans create_frame_x_graph
    index = np.linspace(0, frame_a.shape[1]-1, 75).astype(int)
    frame_a_shorten = np.full((75, 960), -1.1)
    frame_d_shorten = np.full((75, 960), -1.1)
    for i in range(75):
        frame_a_shorten[i,:] = frame_a[:,index[i]]
        frame_d_shorten[i,:] = frame_d[:,index[i]]

    # Créer un tableau de 75 colonnes et 6 lignes
    frame_a_bins = np.zeros((9, 75))  # Increase the size of the array to accommodate the new categories
    for i in range(0, 75, 1):
        for j in range(0, len(frame_a_shorten[i]), 1):
            if frame_a_shorten[i, j] == 0 and frame_d_shorten[i, j] == 0:
                frame_a_bins[0, i] += 1
            elif frame_a_shorten[i, j] == 0 and frame_d_shorten[i, j] < 1 and frame_d_shorten[i, j] > 0:
                frame_a_bins[1, i] += 1
            elif frame_a_shorten[i, j] > 0 and frame_d_shorten[i, j] > 0.9 and \
                    function_1(frame_a_shorten[i, j], frame_d_shorten[i, j], 0.1) < 0.1 and frame_d_shorten[i, j] < 1:
                frame_a_bins[2, i] += 1
            elif function_1(frame_a_shorten[i, j], frame_d_shorten[i, j], 0.1) > 0.1 and \
                    function_1(frame_a_shorten[i, j], frame_d_shorten[i, j], 0.9) < 0.9 \
                    and frame_a_shorten[i, j] < frame_d_shorten[i, j] :
                frame_a_bins[3, i] += 1
            elif frame_a_shorten[i, j] == 0 and frame_d_shorten[i, j] == 1:
                frame_a_bins[4, i] += 1
            elif frame_d_shorten[i, j] < 1 and frame_d_shorten[i, j] > 0.9 and \
                    function_1(frame_a_shorten[i, j], frame_d_shorten[i, j], 0.9) > 0.9\
                    and frame_a_shorten[i, j] < 1:
                frame_a_bins[5, i] += 1
            elif frame_a_shorten[i, j] > 0 and frame_a_shorten[i, j] < 1 and frame_d_shorten[i, j] == 1:
                frame_a_bins[6, i] += 1
            elif frame_a_shorten[i, j] == 1 and frame_d_shorten[i, j] == 1:
                frame_a_bins[7, i] += 1
            else:
                frame_a_bins[8, i] += 1

    test = np.sum(frame_a_bins[:, 10])
    print(frame_a_bins[:, 10])
    for i in range(0, 75, 1):
        sum_q = np.sum(frame_a_bins[:, i])
        frame_a_bins[:, i] = frame_a_bins[:, i] / sum_q
    data = frame_a_bins

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

    plt.show()
    # Sauvegarder le graphique
    plt.savefig('pop_type.png')

    return
# Call the function
#create_frame_x_graph_2()
#create_graph_pop_type_2()
create_frame_x_graph_2()