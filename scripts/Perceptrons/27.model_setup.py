#%%
!pip install sklearn
!pip install torch
import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

#%%
n_points = 100
centers = [[-.5, .5], [.5, -.5]]
X, y = datasets.make_blobs(n_samples=n_points, random_state=123, centers=centers, cluster_std=0.4)
X_data = torch.Tensor(X)
y_data = torch.Tensor(y)

#%%
def scatter_plot():
  plt.scatter(X[y==0, 0], X[y==0, 1])
  plt.scatter(X[y==1, 0], X[y==1, 1])

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


#%%
torch.manual_seed(2)
model = Perceptron(2, 1)

#%%
def get_params():
  [w,b] = model.parameters()
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

