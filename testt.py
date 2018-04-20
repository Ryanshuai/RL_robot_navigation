import torch
import numpy as np
import copy as cp

x = torch.Tensor([1, 2, 3, 4])
print(x)

y = x.squeeze(1)
print(y)