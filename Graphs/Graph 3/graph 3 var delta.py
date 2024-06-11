import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm

# Définir la fonction
def f(a, g,delta, x_tilde, b):
    term1 =(1 - a) * (delta * (1 - b * (1 - a)) * (x_tilde * (1 - a) * (1 - delta) + a) +
                       (1 - b * delta * (1 - a)) * (x_tilde * (1 - delta) + a * delta))
    term2 = (1 - 2 * delta * (1 - a)) * (a - 1 - delta * (1 - a) * (-g * (1 - a)))
    return term1 + term2


# Paramètres donnés
delta = 0.96
x_tilde = 0.96
b = 2


# Créer une grille de valeurs pour a et d
a = np.linspace(0, 1, 100)
g = np.linspace(0, 1, 100)
a, g = np.meshgrid(a, g)

# Calculer les valeurs de la fonction pour chaque point de la grille
z = f(a, g, delta, x_tilde, b)

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
contour = plt.contourf(a, g, z, levels=100, cmap=newcmp, norm=divnorm)
plt.colorbar(contour, label='f(a, d)')
plt.xlabel('a')
plt.ylabel('d')


# Ajouter la courbe de niveau pour f(a, d) = 0
zero_contour = plt.contour(a, g, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, delta = 0.96', inline=True)

delta = 0.03
x_tilde = 0.96
b = 2
z = f(a, g, delta, x_tilde, b)

zero_contour = plt.contour(a, g, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, delta = 0.03', inline=True)

delta = 0.3
x_tilde = 0.96
b = 4
z = f(a, g, delta, x_tilde, b)

zero_contour = plt.contour(a, g, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, delta = 0.3', inline=True)

delta = 0.7
x_tilde = 0.96
b = 4
z = f(a, g, delta, x_tilde, b)

zero_contour = plt.contour(a, g, z, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt='f(a,d)=0, delta = 0.7', inline=True)

plt.show()


