import torch

# Paramètres
number_groups = 3
group_size = 5
device = 'cpu'

# Exemple de tenseurs représentant les joueurs dans chaque groupe
x_i = torch.arange(number_groups * group_size).reshape(number_groups, group_size).to(device)
d_i = torch.arange(number_groups * group_size).reshape(number_groups, group_size).to(device)
a_i = torch.arange(number_groups * group_size).reshape(number_groups, group_size).to(device)

# Affichage des tenseurs avant le mélange
print("Avant le mélange:")
print("x_i:", x_i)
print("d_i:", d_i)
print("a_i:", a_i)

# Mélange des joueurs au sein de chaque groupe en place
rand_values = torch.rand(number_groups, group_size, device=device)
indices = rand_values.argsort(dim=1)
x_i.copy_(x_i.gather(1, indices))
d_i.copy_(d_i.gather(1, indices))
a_i.copy_(a_i.gather(1, indices))

# Affichage des tenseurs après le mélange
print("Après le mélange:")
print("x_i:", x_i)
print("d_i:", d_i)
print("a_i:", a_i)