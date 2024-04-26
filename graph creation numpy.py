import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

def create_frame_x_graph_2():
    # Import frame_x from csv file
    frame_x = np.load('frame_x.npy')

            # Number of desired columns
    target_n_cols = 75

    # Division of columns into 75 groups
    index = np.linspace(0, frame_x.shape[1]-1, target_n_cols).astype(int)
    # Create an empty dataframe for frame_x_shorten
    frame_x_shorten = np.full((target_n_cols, 960), -1.1)

    # For each group
    for i in range(target_n_cols):
        frame_x_shorten[i, :] = frame_x[:, index[i]]
    # create data frame of 10 lines and 10 columns
    frame_x_bins = np.zeros((10, 75))

    bins = np.arange(0, 1.1, 0.1)
    for i in range(0, 75, 1):
        hist, bin_edges = np.histogram(frame_x[:, i], bins)
        frame_x_bins[:, i] = hist

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
    frame_a_bins = np.zeros((6, 75))
    for i in range(0, 75, 1):
        for j in range(0, len(frame_a_shorten[i]), 1):
            if frame_a_shorten[i,j] == 0 and frame_d_shorten[i,j] == 0:
                frame_a_bins[0, i] += 1
            elif frame_a_shorten[i,j] == 0 and frame_d_shorten[i,j] < 1 and frame_d_shorten[i,j] > 0 and frame_d_shorten[i,j] != 1 and frame_a_shorten[i,j] != 1:
                frame_a_bins[1, i] += 1
            elif frame_a_shorten[i,j] == 0 and frame_d_shorten[i,j] == 1:
                frame_a_bins[2, i] += 1
            elif frame_a_shorten[i,j] > 0 and frame_a_shorten[i,j] < 1 and frame_d_shorten[i,j] <1 and frame_d_shorten[i,j] > 0:
                frame_a_bins[3, i] += 1
            elif frame_a_shorten[i,j] > 0 and frame_d_shorten[i,j] == 1 and frame_a_shorten[i,j] < 1:
                frame_a_bins[4, i] += 1
            elif frame_a_shorten[i,j] == 1 and frame_d_shorten[i,j] == 1:
                frame_a_bins[5, i] += 1

    print(frame_a_bins)
    for i in range(0, 75, 1):
        sum_q = np.sum(frame_a_bins[:, i])
        frame_a_bins[:, i] = frame_a_bins[:, i] / sum_q
    data = frame_a_bins

    print(data)
    colors = ['red', 'lightgreen', 'green', 'blue', 'orange', 'brown']
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
    plt.legend(["Unconditionally selfish" ,"De−escalators","Perfect reciprocators","Ambiguous","Escalators","Unconditionally generous"], loc="upper left", bbox_to_anchor=(1,1))

    plt.show()
    # Sauvegarder le graphique
    plt.savefig('pop_type.png')

    return
# Call the function
create_frame_x_graph_2()
create_graph_pop_type_2()
