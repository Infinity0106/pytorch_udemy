#%%
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
import torch.nn as nn
import torch
!pip install sklearn
!pip install torch

#%%
n_points = 500
X, y = datasets.make_circles(
    n_samples=n_points, random_state=123, noise=0.1, factor=0.2)
X_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(500, 1))

#%%


def scatter_plot():
  plt.scatter(X[y == 0, 0], X[y == 0, 1])
  plt.scatter(X[y == 1, 0], X[y == 1, 1])


#%%
scatter_plot()

#%%


class Perceptron(nn.Module):
  def __init__(self, input, hidden1, output):
    super().__init__()
    self.linear = nn.Linear(input, hidden1)
    self.linear2 = nn.Linear(hidden1, output)

  def forward(self, x):
    x = torch.sigmoid(self.linear(x))
    x = torch.sigmoid(self.linear2(x))
    return x

  def predict(self, x):
    pred = self.forward(x)
    return 1 if pred > 0.5 else 0


#%%
torch.manual_seed(2)
model = Perceptron(2, 4, 1)
print(list(model.parameters()))

#%%
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

#%%
epochs = 1000
losses = []
for i in range(epochs):
  y_pred = model.forward(X_data)
  loss = criterion(y_pred, y_data)
  print("{} {}".format(i, loss.item()))

  losses.append(loss)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

#%%
plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("Epochs")
