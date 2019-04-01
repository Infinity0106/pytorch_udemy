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

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])


