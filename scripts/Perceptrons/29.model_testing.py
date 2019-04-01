#%%
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
import torch.nn as nn
import torch
!pip install sklearn
!pip install torch

#%%
n_points = 100
centers = [[-.5, .5], [.5, -.5]]
X, y = datasets.make_blobs(
    n_samples=n_points, random_state=123, centers=centers, cluster_std=0.4)
X_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(100, 1))

#%%


def scatter_plot():
  plt.scatter(X[y == 0, 0], X[y == 0, 1])
  plt.scatter(X[y == 1, 0], X[y == 1, 1])


#%%
scatter_plot()

#%%


class Perceptron(nn.Module):
  def __init__(self, input, output):
    super().__init__()
    self.linear = nn.Linear(input, output)

  def forward(self, x):
    pred = torch.sigmoid(self.linear(x))
    return pred
  
  def predict(self, x):
    pred = torch.sigmoid(self.linear(x))
    return 1 if pred > 0.5 else 0


#%%
torch.manual_seed(2)
model = Perceptron(2, 1)

#%%


def get_params():
  [w, b] = model.parameters()
  w1, w2 = w.view(2)
  b1 = b[0]
  return (w1.item(), w2.item(), b1.item())

#%%


def plot_fit(title):
  plt.title = title
  w1, w2, b1 = get_params()
  x1 = np.array([-2.0, 2.0])
  x2 = (w1*x1 + b1)/-w2
  plt.plot(x1, x2, 'r')
  scatter_plot()


#%%
plot_fit("untrained data")


#%%
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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

#%%
point1 = torch.Tensor([1.0, -1.0])
point2 = torch.Tensor([-1.0, 1.0])
plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
plt.plot(point2.numpy()[0], point2.numpy()[1], 'wo')
print("Redpoint positive probabilitiy = {}".format(model.forward(point1).item()))
print("Blackpoint positive probabilitiy = {}".format(model.forward(point2).item()))
print("Redpoint positive probabilitiy = {}".format(model.predict(point1)))
print("Blackpoint positive probabilitiy = {}".format(model.predict(point2)))
plot_fit("Trained model")
