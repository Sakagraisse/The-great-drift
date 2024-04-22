import pandas as pd
import numpy as np
import random as rd


# take aletory a continuous number between 0 and 1
print(len(np.linspace(0, 2000, 75+1).astype(int)))

print(np.linspace(0, 2000, 75+1).astype(int))

# create data frame of 10 lines and 10 columns
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

# calculate the index of groups to have 75 groups of the same size
index = np.linspace(0, frame_x.shape[1], 75 + 1).astype(int)
# for i in range size of columns frame_x
for i in range(0, 75, 1):
    for j in range(index[i], index[i + 1], 1):
        # append the values of the columns of frame_x to the columns of frame_x_shorten
        frame_x_shorten.iloc[:, i] += frame_x.iloc[:, j]

print(frame_x_shorten.shape)



def create_frame_x_graph():
    # Import frame_x from csv file
    frame_x = pd.read_csv("frame_x.csv")

    # Create an empty list for frame_x_shorten
    frame_x_shorten = []

    # Calculate the index of groups to have 75 groups of the same size
    index = np.linspace(0, frame_x.shape[1], 75 + 1).astype(int)

    # For each group
    for i in range(75):
        # Get the block from frame_x
        block = frame_x.iloc[:, index[i]:index[i + 1]].values.flatten().tolist()

        # Extend frame_x_shorten with the block
        frame_x_shorten += block

    print(len(frame_x_shorten[0:]))  # Print the first 10 elements

# Call the function
create_frame_x_graph()


# Call the function
create_frame_x_graph()