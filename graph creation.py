import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

def create_frame_x_graph():
    # Import frame_x from csv file
    frame_x = pd.read_csv("frame_x.csv")

    # Number of desired columns
    target_n_cols = 75

    # Division of columns into 75 groups
    index = np.linspace(0, frame_x.shape[1], 75 + 1).astype(int)

    # calculate the size between each group
    size = [0]*75
    for i in range(75):
        size[i] = index[i+1] - index[i]
    max = np.max(size)*960

    # Create an empty dataframe for frame_x_shorten
    frame_x_shorten = np.full((75, max), -1.1)



    # For each group
    for i in range(75):
        counter = 0
        for j in range(index[i], index[i + 1], 1):
            frame_x_shorten[i,counter:(counter+960)] = frame_x.iloc[:,j].values
            counter += 960

    # create data frame of 10 lines and 10 columns
    frame_x_bins = pd.DataFrame(np.zeros((10, 75)))

    bins = np.arange(0, 1.1, 0.1)
    for i in range(0, 75, 1):
        hist, bin_edges = np.histogram(frame_x.iloc[:, i], bins)
        frame_x_bins.iloc[:, i] = hist

    data = frame_x_bins.to_numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='Greys', aspect='auto')
    plt.colorbar(label='Value')
    plt.xticks(range(frame_x_bins.shape[1]), frame_x_bins.columns, rotation=0)
    plt.yticks(range(frame_x_bins.shape[0]), bins[:-1])
    plt.gca().invert_yaxis()  # Invert y-axis
    plt.xlabel('Period')
    plt.ylabel('Bins')
    plt.title('Graph')
    plt.show()


def create_graph_pop_type():
    # import frame_a from csv file
    frame_a = pd.read_csv("frame_a.csv")
    # import frame_d from csv file
    frame_d = pd.read_csv("frame_d.csv")

    #create groups of 75 columns like in create_frame_x_graph
    index = np.linspace(0, frame_a.shape[1], 75 + 1).astype(int)
    size = [0]*75
    for i in range(75):
        size[i] = index[i+1] - index[i]
    max = np.max(size)*960
    frame_a_shorten = np.full((75, max), -1.1)
    frame_d_shorten = np.full((75, max), -1.1)
    for i in range(75):
        counter = 0
        for j in range(index[i], index[i + 1], 1):
            frame_a_shorten[i,counter:(counter+960)] = frame_a.iloc[:,j].values
            frame_d_shorten[i,counter:(counter+960)] = frame_d.iloc[:,j].values
            counter += 960



    #create a dataframe of 75 columns and 5 rows
    frame_a_bins = pd.DataFrame(np.zeros((6, 75)))
    for i in range(0, 75, 1):
        for j in range(0, len(frame_a_shorten[i]), 1):
            if frame_a_shorten[i,j] == 0 and frame_d_shorten[i,j] == 0:
                frame_a_bins.iloc[0, i] += 1
            elif frame_a_shorten[i,j] == 0 and frame_d_shorten[i,j] < 1 and frame_d_shorten[i,j] > 0 and frame_d_shorten[i,j] != 1 and frame_a_shorten[i,j] != 1:
                frame_a_bins.iloc[1, i] += 1
            elif frame_a_shorten[i,j] == 0 and frame_d_shorten[i,j] == 1:
                frame_a_bins.iloc[2, i] += 1
            elif frame_a_shorten[i,j] > 0 and frame_a_shorten[i,j] < 1 and frame_d_shorten[i,j] <1 and frame_d_shorten[i,j] > 0:
                frame_a_bins.iloc[3, i] += 1
            elif frame_a_shorten[i,j] > 0 and frame_d_shorten[i,j] == 1 and frame_a_shorten[i,j] < 1:
                frame_a_bins.iloc[4, i] += 1
            elif frame_a_shorten[i,j] == 1 and frame_d_shorten[i,j] == 1:
                frame_a_bins.iloc[5, i] += 1

    print(frame_a_bins)
    for i in range(0, 75, 1):
        sum_q =0
        for j in range(0, 6, 1):
            sum_q += frame_a_bins.iloc[j, i]
        for j in range(0, 6, 1):
            frame_a_bins.iloc[j, i] = frame_a_bins.iloc[j, i]/sum_q
    data = frame_a_bins.to_numpy()
    data_df = pd.DataFrame(data)
    data_df= data_df.transpose()
    print(data_df)
    colors = ['red', 'lightgreen', 'green', 'blue', 'orange', 'brown']
    # Number of groups (i.e., number of stacked bars)
    data_df.plot(
        kind='bar',
        stacked=True,
        color=colors,
        title='Stacked Bar Graph',
        mark_right=True,
        figsize = (12, 8))
    plt.subplots_adjust(right=0.8)
    plt.xlabel("Generation")
    plt.ylabel("Proportion")
    plt.legend(["Unconditionally selfish" ,"De−escalators","Perfect reciprocators","Ambiguous","Escalators","Unconditionally generous"], loc="upper left", bbox_to_anchor=(1,1))

    plt.show()

    return
# Call the function
#create_frame_x_graph()
create_graph_pop_type()
