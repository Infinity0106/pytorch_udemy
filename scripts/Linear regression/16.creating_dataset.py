# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
!pip install torch

# %%
X = torch.randn(100, 1) * 10
y = X + 3 * torch.rand(100, 1)
plt.plot(X.numpy(), y.numpy(), 'o')
plt.ylabel("Y")
plt.xlabel("X")


# %%
class LR(nn.Module):  # inheritance nn.module
    def __init__(self, input_size, output_size):
        super().__init__()  # intilize parent class
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred


# %%
torch.manual_seed(1)
model = LR(1, 1)

# %%
x = torch.tensor([[1.0], [2.0]])
model.forward(x)