import pandas as pd
import numpy as np


def create_frame_x_graph():
    # Import frame_x from csv file
    frame_x = pd.read_csv("frame_x.csv")

    # Number of desired columns
    target_n_cols = 75

    # Division of columns into 75 groups
    index = np.linspace(0, frame_x.shape[1], num=target_n_cols + 1, dtype=int)

    # Prepare an empty DataFrame for the shortened frame_x
    frame_x_shorten = pd.DataFrame(index=frame_x.index)

    # For each group of columns, concatenate them into a single column
    for i in range(target_n_cols):
        # Extract part of the frame relevant to the current group
        current_group = frame_x.iloc[:, index[i]:index[i + 1]]

        # Concatenate all columns in the current group into a single string column
        # If the columns are not strings, you need to convert them first
        frame_x_shorten[f'col_{i}'] = current_group.apply(lambda x: ''.join(x.astype(str)), axis=1)

    return frame_x_shorten


# Example to use this function
shortened_df = create_frame_x_graph()


