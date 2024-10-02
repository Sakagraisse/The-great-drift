import torch

# Tensor initial
tensor = torch.tensor([1, 2, 3])

# Nouveau tensor à ajouter
new_tensor = torch.tensor([4])

# Concaténation des deux tenseurs
tensor = torch.cat((tensor, new_tensor))

print(tensor)
