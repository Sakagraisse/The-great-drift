import matplotlib.pyplot as plt
import numpy as np
import os

# Create a figure and axis
fig, ax = plt.subplots(figsize=(11, 6))

# Define the function for each region boundary
x = np.linspace(0, 1, 400)



# De-escalators: a = 0, d in (0, 1)

ax.axhline(y=0.002, color='red', alpha=1, linewidth=4, label='Unconditionally selfish')

ax.plot(x, (x*0.5), color='lightgreen', alpha=1, linewidth=4, label='De-escalators')


ax.plot(x, (0.3 + x*(0.96 - 0.3)), color='purple', alpha=1, linewidth=4,label='Quasi-de-escalators')

ax.plot(x, (0.3 + x*(0.7 - 0.3)), color='blue', alpha=1, linewidth=4,label='Ambiguous')

ax.plot(x, x, color='green', alpha=1, linewidth=4, label='Perfect reciprocators')

ax.plot(x, (0.05 + x*(0.7 - 0.03)), color='lightblue', alpha=1, linewidth=4, label='Quasi-escalators')

ax.plot(x, (x*0.5+0.5 ), color='orange', alpha=1, linewidth=4, label='Escalators')



ax.axhline(y=0.998, color='brown', alpha=1, linewidth=4, label='Unconditionally generous')






# Label the axes and set limits


ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Make the numbers on the axes bigger
ax.tick_params(axis='both', labelsize=15)  # Change the size of the numbers on the x and y axes to 15

# Create a secondary y-axis and set its label to 'd'
ax2 = ax.twinx()
ax2.set_ylabel('d')

ax2.tick_params(axis='both', labelsize=15)  # Change the size of the numbers on the x and y axes to 15

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax.set_ylabel('a', fontsize=20, rotation=0)
ax.yaxis.set_label_coords(-0.07, 0.45)

# Set the y-label for the right y-axis, increase its size, adjust its position, and rotate it by 90 degrees
ax2.set_ylabel('d', fontsize=20, rotation=0)
ax2.yaxis.set_label_coords(1.07, 0.5)

# Move the legend to the upper right corner outside the plot
ax.legend(["Unconditionally \nselfish", "De-escalators", "Quasi-\nde-escalator", "Ambiguous", "Perfect \nreciprocators",
                "Quasi-\nescalator", "Escalators",
                "Unconditionally \ngenerous"], bbox_to_anchor=(1.11, 1), fontsize=11.5)

plt.subplots_adjust(left=0.07, right=0.73, top=0.95,bottom=0.05)

dir_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(dir_path, 'type_of_player_quasi.pdf')
plt.savefig(file_path)
