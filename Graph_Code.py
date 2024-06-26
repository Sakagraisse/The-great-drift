import os
import numpy as np
import matplotlib.pyplot as plt
import numba as nb


@nb.jit(nopython=True)
def compute_graph_1(frame_x):
    """Compute the density values for the first graph from the frame_x array"""
    frame_x_bins = np.zeros((76, 10))
    bins = np.arange(0, 1.1, 0.1)
    for i in range(0, 76, 1):
        hist, bin_edges = np.histogram(frame_x[i, :], bins)
        frame_x_bins[i, :] = hist

    data = frame_x_bins
    data = data.T

    return data

def create_graph_1(period,figsize):
    """ Create the first graph showing the evolution of the first move """
    # Import frame_x from npy file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_x = np.load(os.path.join(dir_path, 'frame_x.npy'))

    # Compute the density for the graph
    data = compute_graph_1(frame_x)

    # Create a figure
    fig1 = plt.figure(figsize=figsize)
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
    plt.title('Evolution of First Move', fontsize=20, fontweight='bold')

    # Adjust layout for better visualization
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)

    # Return the figure
    return fig1

@nb.jit(nopython=True)
def function_1(a,b):
    """function to compute the type of ambigous strategies"""
    y = a/(a+1-b)
    return y

@nb.jit(nopython=True)
def compute_graph_2(frame_a, frame_d):
    """Compute the barchart values for the second graph from the frame_a and frame_d arrays"""
    # Create an array to store the values of the barchart
    frame_a_bins = np.zeros((76, 9))

    # Loop over the generations
    for j in range(0, 76, 1):
        # Loop over the individuals to find the type of strategy
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

    for i in range(0, 76, 1):
        sum_q = np.sum(frame_a_bins[i, :])
        frame_a_bins[i, :] = frame_a_bins[i, :] / sum_q
    data = frame_a_bins

    #transposing the data
    data = data.T

    return data


def create_graph_2(period,figsize):
    """Create the second graph showing the evolution of the strategies"""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_a = np.load(os.path.join(dir_path, 'frame_a.npy'))
    frame_d = np.load(os.path.join(dir_path, 'frame_d.npy'))

    data = compute_graph_2(frame_a, frame_d)
    colors = ['red', 'lightgreen', 'purple', 'blue', 'darkgreen', 'lightblue', 'orange', 'brown', 'grey']
    fig2 = plt.figure(figsize=figsize)
    plt.bar(np.arange(data.shape[1]), data[0, :], color=colors[0])
    bottom = data[0, :]
    for i in range(1, data.shape[0]):
        plt.bar(np.arange(data.shape[1]), data[i, :], bottom=bottom, color=colors[i])
        bottom += data[i, :]
    # Customizing x-axis labels
    custom_ticks = np.arange(0, 76, 25)
    custom_labels = [f'Gen {int((i/75)*period)}' for i in custom_ticks]
    plt.xticks(custom_ticks, custom_labels, rotation=45)

    plt.gca().yaxis.set_tick_params(rotation=90)

    plt.title('Evolution of Strategies', fontsize=20, fontweight='bold')
    plt.xlabel("Generation", fontsize=15, fontweight='bold')
    plt.ylabel("Proportion", fontsize=15, fontweight='bold')

    # Add a legend
    plt.legend(["Unconditionally selfish", "De-escalators", "Quasi-de-escalator", "Ambiguous", "Perfect reciprocators", "Quasi-escalator", "Escalators",
                "Unconditionally generous", "Other"], bbox_to_anchor=(1.05, 1), fontsize='x-small')


    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.2)

    return fig2


def create_graph_3(period,figsize):
    """Create the third graph showing the surplus per generation"""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_surplus = np.load(os.path.join(dir_path, 'frame_surplus.npy'))
    #sum for each line
    frame_surplus = np.mean(frame_surplus, axis=1)
    #create a graph bar showing the surplus
    fig3 = plt.figure(figsize=figsize)
    plt.bar(np.arange(frame_surplus.shape[0]), frame_surplus)

    # Customize x-axis labels
    custom_ticks = np.arange(0, 76, 25)
    custom_labels = [f'Gen {int((i/75)*period)}' for i in custom_ticks]
    plt.xticks(custom_ticks, custom_labels, rotation=45)

    # Labels and title
    plt.xlabel('Generation', fontsize=15, fontweight='bold')
    plt.ylabel('Surplus', fontsize=15, fontweight='bold')
    plt.title('Surplus per generation', fontsize=20, fontweight='bold')

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)

    return fig3

def store_all_graphs(period):
    """Regenerate and Store all the graphs in a pdf file"""
    # Import frame_x from npy file
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # generate the first graph
    frame_x = np.load(os.path.join(dir_path, 'frame_x.npy'))
    data = compute_graph_1(frame_x)

    # Create a figure
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='gray_r', aspect='auto')

    # Customize x-axis labels
    custom_ticks = np.arange(0, 76, 25)
    custom_labels = [f'Gen {int((i / 75) * period)}' for i in custom_ticks]
    plt.xticks(custom_ticks, custom_labels, rotation=0)

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
    plt.gca().yaxis.set_tick_params(rotation=0)
    plt.tick_params(axis='x', labelsize=15)  # Change the size of the numbers on the x-axis to 10
    plt.tick_params(axis='y', labelsize=15)  # Change the size of the numbers on the y-axis to 10

    # Labels and title
    plt.xlabel('Generation', fontsize=20, fontweight='bold')
    plt.ylabel('Proportion', fontsize=20, fontweight='bold')

    # Adjust layout for better visualization
    plt.subplots_adjust(left=0.08, right=1, top=0.98,bottom=0.11)
    # Add colorbar outside the plot
    plt.colorbar()
    # Save the figure
    plt.savefig(os.path.join(dir_path, 'Evolution_of_First_Move.pdf'), dpi=150)

    #graph 2
    frame_a = np.load(os.path.join(dir_path, 'frame_a.npy'))
    frame_d = np.load(os.path.join(dir_path, 'frame_d.npy'))
    data = compute_graph_2(frame_a, frame_d)

    colors = ['red', 'lightgreen', 'purple', 'blue', 'darkgreen', 'lightblue', 'orange', 'brown', 'grey']
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(data.shape[1]), data[0, :], color=colors[0])
    bottom = data[0, :]
    for i in range(1, data.shape[0]):
        plt.bar(np.arange(data.shape[1]), data[i, :], bottom=bottom, color=colors[i])
        bottom += data[i, :]
    custom_ticks = np.arange(0, 76, 25)
    custom_labels = [f'Gen {int((i / 75) * period)}' for i in custom_ticks]
    plt.xticks(custom_ticks, custom_labels, rotation=0)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.gca().yaxis.set_tick_params(rotation=0)
    plt.xlabel("Generation", fontsize=20, fontweight='bold')
    plt.ylabel("Proportion", fontsize=20, fontweight='bold')

    plt.legend(["Unconditionally \nselfish", "De-escalators", "Quasi-\nde-escalator", "Ambiguous", "Perfect \nreciprocators",
                "Quasi-\nescalator", "Escalators",
                "Unconditionally \ngenerous", "Other"], bbox_to_anchor=(1, 1), fontsize=11.5)

    plt.subplots_adjust(left=0.08, right=0.8, top=0.98,bottom=0.11)
    plt.savefig(os.path.join(dir_path, 'Evolution_of_Strategies.pdf'), dpi=150)

    #graph 3
    dir_path = os.path.dirname(os.path.abspath(__file__))
    frame_surplus = np.load(os.path.join(dir_path, 'frame_surplus.npy'))
    # sum for each line
    frame_surplus = np.mean(frame_surplus, axis=1)
    # create a graph bar showing the surplus
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(frame_surplus.shape[0]), frame_surplus)

    # Customize x-axis labels
    custom_ticks = np.arange(0, 76, 25)  # Example: custom ticks every 25 periods
    custom_labels = [f'Gen {int((i / 75) * period)}' for i in custom_ticks]  # Example: custom labels
    plt.xticks(custom_ticks, custom_labels, rotation=0)

    # Labels and title
    plt.xlabel('Generation', fontsize=20, fontweight='bold')
    plt.ylabel('Surplus', fontsize=20, fontweight='bold')
    plt.subplots_adjust(left=0.08, right=0.8, top=0.98,bottom=0.11)
    # Sauvegarder le graphique
    plt.savefig(os.path.join(dir_path, 'Surplus_per_generation.pdf'), dpi=150)


    return

