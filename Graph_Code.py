import sys
import os


import numpy as np

import matplotlib.pyplot as plt
import numba as nb


# calaculus for graph 1

@nb.jit(nopython=True)
def compute_graph_1(frame_x):

    frame_x_bins = np.zeros((75, 10))
    bins = np.arange(0, 1.1, 0.1)
    for i in range(0, 75, 1):
        hist, bin_edges = np.histogram(frame_x[i, :], bins)
        frame_x_bins[i, :] = hist

    data = frame_x_bins
    # transposing the data
    # Transposer les données comme dans votre code
    data = data.T

    return data

def create_graph_1(period):
    # Import frame_x from npy file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_x = np.load(os.path.join(dir_path, 'frame_x.npy'))

    data = compute_graph_1(frame_x)

    # Create a figure
    fig = plt.figure()
    plt.imshow(data, cmap='gray_r', aspect='auto')

    # Customize x-axis labels
    custom_ticks = np.arange(0, 76, 25)
    custom_labels = [f'Gen {int((i/75)*period)}' for i in custom_ticks]
    plt.xticks(custom_ticks, custom_labels, rotation=45)

    # Customize y-axis labels
    custom_ticks = np.arange(0, 1.1, 0.1)
    custom_ticks = np.round(custom_ticks, 1)
    plt.yticks(np.linspace(0, data.shape[0] - 1, len(custom_ticks)), custom_ticks)
    plt.gca().invert_yaxis()

    # Customize y-axis labels
    num_ticks = data.shape[0] + 1
    custom_ticks = np.linspace(0, 1, num_ticks)
    custom_ticks = np.round(custom_ticks, 2)
    plt.yticks(np.arange(-0.5, data.shape[0]), custom_ticks)


    # Customize y-axis tick rotation
    plt.gca().yaxis.set_tick_params(rotation=90)

    # Labels and title
    plt.xlabel('Generation', fontsize=15, fontweight='bold')
    plt.ylabel('Proportion', fontsize=15, fontweight='bold')
    plt.title('Coupled', fontsize=20, fontweight='bold')

    # Adjust layout for better visualization
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    # Return the figure
    return fig

@nb.jit(nopython=True)
def function_1(a,b):
    y = a/(a+1-b)
    return y

@nb.jit(nopython=True)
def compute_graph_2(frame_a, frame_d):

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

    print(frame_a_bins.shape)
    for i in range(0, 75, 1):
        sum_q = np.sum(frame_a_bins[i, :])
        frame_a_bins[i, :] = frame_a_bins[i, :] / sum_q
    data = frame_a_bins
    #transposing the data
    data = data.T

    return data


def create_graph_2(period):
    # Importer frame_a et frame_d à partir de fichiers npy
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_a = np.load(os.path.join(dir_path, 'frame_a.npy'))
    frame_d = np.load(os.path.join(dir_path, 'frame_d.npy'))

    data = compute_graph_2(frame_a, frame_d)

    colors = ['red', 'lightgreen', 'purple', 'blue', 'darkgreen', 'lightblue', 'orange', 'brown', 'grey']
    # Nombre de groupes (c'est-à-dire nombre de barres empilées)
    fig, ax = plt.subplots()
    plt.bar(np.arange(data.shape[1]), data[0, :], color=colors[0])
    bottom = data[0, :]
    for i in range(1, data.shape[0]):
        plt.bar(np.arange(data.shape[1]), data[i, :], bottom=bottom, color=colors[i])
        bottom += data[i, :]
    # Customiser les étiquettes de l'axe des x
    custom_ticks = np.arange(0, 76, 25)  # Example: custom ticks every 25 periods
    custom_labels = [f'Gen {int((i/75)*period)}' for i in custom_ticks]  # Example: custom labels
    plt.xticks(custom_ticks, custom_labels, rotation=45)

    plt.gca().yaxis.set_tick_params(rotation=90)

    plt.title('Stacked Bar Graph', fontsize=20, fontweight='bold')
    plt.xlabel("Generation", fontsize=15, fontweight='bold')
    plt.ylabel("Proportion", fontsize=15, fontweight='bold')

    # Ajuster la position de la légende
    plt.legend(["Unconditionally selfish", "De-escalators", "Quasi-de-escalator", "Ambiguous", "Perfect reciprocators", "Quasi-de-escalator", "Escalators",
                "Unconditionally generous", "Other"], loc="upper left", bbox_to_anchor=(1.05, 1))



    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.15)  # Ajuster la mise en page pour laisser de la place pour la légende

    return fig










def store_all_graphs(period):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_x = np.load(os.path.join(dir_path, 'frame_x.npy'))
    data = compute_graph_1(period,frame_x)

    # Création du graphique
    plt.figure(figsize=(12, 8))
    plt.imshow(data, cmap='viridis', aspect='auto')  # Utilisation de 'viridis' pour une meilleure lisibilité
    plt.colorbar(label='Value')

    # Calcul des indices pour le début, un quart, trois quarts et la fin
    period = data.shape[1]
    indices = [0, period // 3, 2 * period // 3, period - 1]

    # Sélection des périodes correspondantes
    selected_periods = np.arange(period)
    selected_periods = selected_periods[indices]

    # Définir les ticks sur l'axe des x aux indices sélectionnés
    plt.xticks(indices, selected_periods)
    custom_ticks = np.arange(0, 1.1, 0.1)  # définir les ticks personnalisés pour l'axe y
    custom_ticks = np.round(custom_ticks, 1)  # arrondir à une décimale

    # Définir les ticks sur l'axe des y
    plt.yticks(np.linspace(0, data.shape[0] - 1, len(custom_ticks)), custom_ticks)
    plt.gca().invert_yaxis()  # Inverser l'axe y pour correspondre au graphique fourni

    # Étiquettes et titre
    plt.xlabel('Generation')
    plt.ylabel('Proportion')
    plt.title('Coupled $m_j=0$')
    # plt.show()
    # save the graph
    dir_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dir_path, 'frame_x.png'), dpi=75)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_a = np.load(os.path.join(dir_path, 'frame_a.npy'))
    frame_d = np.load(os.path.join(dir_path, 'frame_d.npy'))


    frame_a_bins = compute_graph_2(frame_a, frame_d)

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


