import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import os

def f(h, g, delta, x_tilde, b):
    numerator = (1 + delta - 2 * b * delta) * ((1 - h) * x_tilde + g * (x_tilde - 1))
    denominator = 2 * (1 - delta) * (1 - delta * (h - g))
    result = numerator / denominator
    return result


# Set the base parameters for the graph
delta = 0.96
x_tilde = 0.96
b = 2


# Create a grid of points
h = np.linspace(0, 1, 100)
g = np.linspace(0, 1, 100)
h, g = np.meshgrid(h, g)

# Calculate the function values at each point
z = f(h, g, delta, x_tilde, b)

# Create a colormap for positive and negative values
positive_cm = plt.cm.Blues_r(np.linspace(0.5, 1, 128))
negative_cm = plt.cm.Reds_r(np.linspace(0, 0.5, 128))

# Combine the two colormaps
newcolors = np.vstack((negative_cm, positive_cm))
newcmp = ListedColormap(newcolors)

# Create a diverging color normalization
divnorm = TwoSlopeNorm(vmin=z.min(), vcenter=0, vmax=z.max())

# Create the figure and axis
plt.figure(figsize=(8, 6))
contour = plt.contourf(h, g, z, levels=100, cmap=newcmp, norm=divnorm)
plt.colorbar(contour, label='f(a, d)')
plt.xlabel('a')
plt.ylabel('d')


zero_contour = plt.contour(h, g, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, x_tilde = 2', inline=True)

# Add a contour line for x_tilde = 0.04
delta = 0.96
x_tilde = 0.04
b = 2
z = f(h, g, delta, x_tilde, b)


zero_contour = plt.contour(h, g, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, x_tilde = 0.04', inline=True)

# Add a contour line for x_tilde = 0.3
delta = 0.96
x_tilde = 0.3
b = 2
z = f(h, g, delta, x_tilde, b)


zero_contour = plt.contour(h, g, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, x_tilde= 0.3', inline=True)

# Add a contour line for x_tilde = 0.7
delta = 0.96
x_tilde = 0.7
b = 2
z = f(h, g, delta, x_tilde, b)

zero_contour = plt.contour(h, g, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, x_tilde = 0.7', inline=True)
plt.tick_params(axis='both', which='major', labelsize=14)

# Increase the size of the axis labels
plt.xlabel('a', fontsize=16)
plt.ylabel('d', fontsize=16)
dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the path to the directory where you want to save the file
save_dir = os.path.join(dir_path, 'Graphs')

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save the figure
plt.savefig(os.path.join(save_dir, 'g1 v x.pdf'))
