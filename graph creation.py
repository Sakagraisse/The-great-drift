import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_frame_x_graph():
    #import frame_x from csv file
    frame_x = pd.read_csv("frame_x.csv")

    #create data frame of 10 lines and 10 columns
    frame_x_bins = pd.DataFrame(np.zeros((10, 10)))

    bins = np.arange(0, 1.1, 0.1)
    for i in range(0, 10, 1):
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

# Call the function
create_frame_x_graph()



