import numpy as np
import matplotlib.pyplot as plt
import os

# Définir la fonction
def f(d, delta, x_tilde, b):
    term1 = d * (delta * (1 - b * d) * (x_tilde * d * (1 - delta)) + (1 - b * delta * d) * (x_tilde * (1 - delta)))
    return term1

# Paramètres donnés
delta = 0.96
x_tilde = 0.96

# Plage de valeurs pour d
d_values = np.linspace(0, 1, 400)

# Création du graphique
plt.figure(figsize=(8, 6))

# Fonction pour annoter les lignes
def annotate_line(d_values, f_values, b, ax):
    idx = np.argmax(d_values > 0.8)  # Trouver l'index où d est environ 0.5
    ax.annotate(f'b={b}', xy=(d_values[idx], f_values[idx]), xytext=(5, 5),
                textcoords='offset points', color='black', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

# Calculer et tracer pour b = 2
b = 2
f_values = f(d_values, delta, x_tilde, b)
positive_values = np.ma.masked_less_equal(f_values, 0)
negative_values = np.ma.masked_greater(f_values, 0)
plt.plot(d_values, positive_values, 'b', label='f(d) > 0$')
plt.plot(d_values, negative_values, 'r', label='f(d) < 0$')
annotate_line(d_values, f_values, b, plt.gca())

# Calculer et tracer pour b = 3
b = 3
f_values = f(d_values, delta, x_tilde, b)
positive_values = np.ma.masked_less_equal(f_values, 0)
negative_values = np.ma.masked_greater(f_values, 0)
plt.plot(d_values, positive_values, 'b')
plt.plot(d_values, negative_values, 'r')
annotate_line(d_values, f_values, b, plt.gca())

# Calculer et tracer pour b = 4
b = 4
f_values = f(d_values, delta, x_tilde, b)
positive_values = np.ma.masked_less_equal(f_values, 0)
negative_values = np.ma.masked_greater(f_values, 0)
plt.plot(d_values, positive_values, 'b')
plt.plot(d_values, negative_values, 'r')
annotate_line(d_values, f_values, b, plt.gca())

plt.title('Graph of the function $f(d, \delta, \\tilde{x}, b)$')
plt.xlabel('$d$')
plt.ylabel('$f(d, \delta, \\tilde{x}, b)$')
plt.grid(False)
plt.legend()
# Get the directory of the current script
dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the path to the directory where you want to save the file
save_dir = os.path.join(dir_path, 'Graphs', 'Graph 1')

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save the figure
plt.savefig(os.path.join(save_dir, 'g2 v b.pdf'))
