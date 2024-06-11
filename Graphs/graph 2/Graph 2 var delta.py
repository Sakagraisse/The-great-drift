import numpy as np
import matplotlib.pyplot as plt

# Définir la fonction
def f(d, delta, x_tilde, b):
    term1 = d * (delta * (1 - b * d) * (x_tilde * d * (1 - delta)) + (1 - b * delta * d) * (x_tilde * (1 - delta)))
    return term1

# Paramètres donnés
delta = 0.96
x_tilde = 0.96
b = 2

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
delta = 0.04
f_values = f(d_values, delta, x_tilde, b)
positive_values = np.ma.masked_less_equal(f_values, 0)
negative_values = np.ma.masked_greater(f_values, 0)
plt.plot(d_values, positive_values, 'b', label='f(d) > 0$')
plt.plot(d_values, negative_values, 'r', label='f(d) < 0$')
annotate_line(d_values, f_values, delta, plt.gca())

# Calculer et tracer pour b = 3
delta = 0.3
f_values = f(d_values, delta, x_tilde, b)
positive_values = np.ma.masked_less_equal(f_values, 0)
negative_values = np.ma.masked_greater(f_values, 0)
plt.plot(d_values, positive_values, 'b')
plt.plot(d_values, negative_values, 'r')
annotate_line(d_values, f_values, delta, plt.gca())

# Calculer et tracer pour b = 4
delta = 0.7
f_values = f(d_values, delta, x_tilde, b)
positive_values = np.ma.masked_less_equal(f_values, 0)
negative_values = np.ma.masked_greater(f_values, 0)
plt.plot(d_values, positive_values, 'b')
plt.plot(d_values, negative_values, 'r')
annotate_line(d_values, f_values, delta, plt.gca())

plt.title('Graph of the function $f(d, \delta, \\tilde{x}, b)$')
plt.xlabel('$d$')
plt.ylabel('$f(d, \delta, \\tilde{x}, b)$')
plt.grid(False)
plt.legend()
plt.show()
