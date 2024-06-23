import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import os

# Définir la fonction
def f(a, d, delta, x_tilde, b):
    term1_numerator = a * delta * (1 - b * (d - a)) * (
                (1 - delta * d * (d - a)) - (x_tilde * (d - a) * (1 - delta) + a))
    term2_numerator = a * (1 - b * delta * (d - a)) * ((1 - delta * d * (d - a)) - (x_tilde * (1 - delta) + a * delta))

    denominator = 2 * (1 - delta) * (1 - delta * d * (d - a)) * (1 - delta * (d - a) ** 2)

    term1 = term1_numerator / denominator
    term2 = term2_numerator / denominator

    return term1 + term2

# Paramètres
delta = 0.96
x_tilde = 0.96
b = 2

# Créer une grille de valeurs pour a et d
a = np.linspace(0, 1, 100)
d = np.linspace(0, 1, 100)
a, d = np.meshgrid(a, d)

# Calculer les valeurs de la fonction pour chaque point de la grille
z = f(a, d, delta, x_tilde, b)

# Créer une colormap personnalisée pour les valeurs négatives et positives
positive_cm = plt.cm.Blues_r(np.linspace(0.5, 1, 128))
negative_cm = plt.cm.Reds_r(np.linspace(0, 0.5, 128))

# Fusionner les deux colormaps en une seule
newcolors = np.vstack((negative_cm, positive_cm))
newcmp = ListedColormap(newcolors)

# Créer une norme pour centrer la colormap à zéro
divnorm = TwoSlopeNorm(vmin=z.min(), vcenter=0, vmax=z.max())

# Tracer le graphique
plt.figure(figsize=(8, 6))
contour = plt.contourf(a, d, z, levels=100, cmap=newcmp, norm=divnorm)
plt.colorbar(contour, label='f(a, d)')
plt.xlabel('a')
plt.ylabel('d')

# Ajouter la courbe de niveau pour f(a, d) = 0
zero_contour = plt.contour(a, d, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, x = 0,04', inline=True)

delta = 0.96
x_tilde = 0.04
b = 2
z = f(a, d, delta, x_tilde, b)

zero_contour = plt.contour(a, d, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, x = 0.04', inline=True)

delta = 0.96
x_tilde = 0.3
b = 2
z = f(a, d, delta, x_tilde, b)

zero_contour = plt.contour(a, d, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, x=0.3', inline=True)

delta = 0.96
x_tilde = 0.7
b = 2
z = f(a, d, delta, x_tilde, b)

zero_contour = plt.contour(a, d, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, x=0.7', inline=True)

# Increase the size of the tick labels
plt.tick_params(axis='both', which='major', labelsize=14)

# Increase the size of the axis labels
plt.xlabel('a', fontsize=16)
plt.ylabel('d', fontsize=16)
# Get the directory of the current script
dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the path to the directory where you want to save the file
save_dir = os.path.join(dir_path, 'Graphs', 'Graph 1')

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save the figure
plt.savefig(os.path.join(save_dir, 'g1 v x.pdf'))