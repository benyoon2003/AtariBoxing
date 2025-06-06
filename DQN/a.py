import torch
import numpy as np

a = torch.Tensor([2, 3, 4, 4])
max_indixes = torch.where(a == a.max())[0]
print(np.random.choice(max_indixes))