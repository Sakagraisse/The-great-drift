import torch

group_size = 6
device = 'cpu'

# Calcul du nombre de paires
num_pairs = group_size // 2

# Génération des indices
p1_indices = torch.arange(0, num_pairs * 2, 2, device=device)
p2_indices = torch.arange(1, num_pairs * 2, 2, device=device)

print("Indices des joueurs de la première paire (p1) :", p1_indices)
print("Indices des joueurs de la deuxième paire (p2) :", p2_indices)