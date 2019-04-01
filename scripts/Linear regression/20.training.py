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
[w, b] = model.parameters()


def get_params():
    return (w[0][0].item(), b[0].item())

#%%


def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 = np.array([-30, 30])
    y1 = w1 * x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X, y)


#%%
plot_fit("initial model")

#%%
criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#%%
epochs = 100 #generaciones
losses = []
for i in range(epochs):
  y_pred = model.forward(X)
  loss = criteria(y_pred, y)
  print("{} {}".format(i, loss.item()))
  losses.append(loss)
  optimizer.zero_grad() # reiniciar gradiente
  loss.backward() # derivada
  optimizer.step() # actualizar

#%%
plt.plot(range(epochs), losses)
plt.ylabel("loss")
plt.xlabel("epoch")

#%%
plot_fit("trained model")