import numpy as np
import matplotlib.pyplot as plt

# Create a grid of a and b values
a_values = np.linspace(0, 1, 100)
b_values = np.linspace(0, 1, 100)
a, b = np.meshgrid(a_values, b_values)

# Calculate the expression
with np.errstate(divide='ignore', invalid='ignore'):
    expression = np.divide(a, a + 1 - b)
    expression[(a + 1 - b) == 0] = np.nan  # handle division by zero

# Plot the result as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(expression, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Value of a / (a + 1 - b)')
plt.xlabel('b')
plt.ylabel('a')
plt.title('Heatmap of a / (a + 1 - b)')
plt.grid(True)
plt.show()
