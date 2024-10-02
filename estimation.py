import numpy as np

# Dimensions
group_size = 24
number_groups = 40
num_interactions = 100
period = 100000  # Updated period

# Taille d'un élément en octets (float32)
element_size = 4

# Tenseurs PyTorch
tensor_size = number_groups * group_size * element_size
store_interaction_size = number_groups * group_size * num_interactions * element_size

# Tableaux NumPy
frame_size = period * group_size * number_groups * element_size

# Total RAM en octets
total_ram = 7 * tensor_size + store_interaction_size + 5 * frame_size

# Conversion en Mo
total_ram_mb = total_ram / (1024 ** 2)

print(f"Estimation de la RAM nécessaire par itération : {total_ram_mb:.2f} Mo")