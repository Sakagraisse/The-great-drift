import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import os

def f(a, d, delta, x_tilde, b):
    term1_numerator = a * delta * (1 - b * (d - a)) * (
                (1 - delta * d * (d - a)) - (x_tilde * (d - a) * (1 - delta) + a))
    term2_numerator = a * (1 - b * delta * (d - a)) * ((1 - delta * d * (d - a)) - (x_tilde * (1 - delta) + a * delta))

    denominator = 2 * (1 - delta) * (1 - delta * d * (d - a)) * (1 - delta * (d - a) ** 2)

    term1 = term1_numerator / denominator
    term2 = term2_numerator / denominator

    return term1 + term2

# Set bad values to white
delta = 0.96
x_tilde = 0.96
b = 2

# Create a grid of values for a and d
a = np.linspace(0, 1, 100)
d = np.linspace(0, 1, 100)
a, d = np.meshgrid(a, d)

# Calculate the value of f(a, d) for each pair of a and d
z = f(a, d, delta, x_tilde, b)

# Create a colormap that is white for bad values and blue for good values
positive_cm = plt.cm.Blues_r(np.linspace(0.5, 1, 128))
negative_cm = plt.cm.Reds_r(np.linspace(0, 0.5, 128))

# Combine the two colormaps
newcolors = np.vstack((negative_cm, positive_cm))
newcmp = ListedColormap(newcolors)

# Create a colormap that is white for bad values and blue for good values
divnorm = TwoSlopeNorm(vmin=z.min(), vcenter=0, vmax=z.max())

# Create a contour plot of f(a, d)
plt.figure(figsize=(8, 6))
contour = plt.contourf(a, d, z, levels=100, cmap=newcmp, norm=divnorm)
plt.colorbar(contour, label='f(a, d)')
plt.xlabel('a')
plt.ylabel('d')


# Add a contour line for delta = 0.96
zero_contour = plt.contour(a, d, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, delta = 0,96', inline=True)

# Add a contour line for delta = 0.04
delta = 0.04
x_tilde = 0.96
b = 2
z = f(a, d, delta, x_tilde, b)

zero_contour = plt.contour(a, d, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, delta = 0,04', inline=True)

# Add a contour line for delta = 0.3
delta = 0.3
x_tilde = 0.96
b = 2
z = f(a, d, delta, x_tilde, b)

zero_contour = plt.contour(a, d, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, delta=0,3', inline=True)

# Add a contour line for delta = 0.04
delta = 0.7
x_tilde = 0.96
b = 2
z = f(a, d, delta, x_tilde, b)

zero_contour = plt.contour(a, d, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, delta=0,7', inline=True)

# Increase the size of the tick labels
plt.tick_params(axis='both', which='major', labelsize=14)

# Increase the size of the axis labels
plt.xlabel('a', fontsize=16)
plt.ylabel('d', fontsize=16)
# Get the directory of the current script
dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the path to the directory where you want to save the file
save_dir = os.path.join(dir_path, 'Graphs')

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save the figure
plt.savefig(os.path.join(save_dir, 'g1 v delta.pdf'))