#%%
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
import  torch.nn.functional as F
from matplotlib import pyplot as plt
!pip install torch torchvision

#%%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
training_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(
    dataset=training_dataset, batch_size=100, shuffle=True)

#%%
def image_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image


#%%
dataiter = iter(training_loader)
images, labels = dataiter.next()
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(image_convert(images[idx]))
  ax.set_title([labels[idx].item()])


#%%
class Classifier(nn.Module):
  def __init__(self, d_in, h1, h2, d_out):
    super().__init__()
    self.linear1 = nn.Linear(d_in, h1)
    self.linear2 = nn.Linear(h1, h2)
    self.linear3 = nn.Linear(h2, d_out)

  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    
    return x

  
#%%
model = Classifier(28*28, 125, 65, 10)

#%%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#%%
epochs = 12
running_loss_history = []
running_corrects_history = []

for e in range(epochs):

  running_loss = 0.0
  running_corrects = 0.0
  for inputs, labels in training_loader:
    inputs = inputs.view(inputs.shape[0], -1)
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs,1);
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)
  else:
    epoch_loss = running_loss/len(training_loader)
    epoch_acc = running_corrects.float()/len(training_loader)
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc.item())
    print("Training loss: {:.4f}, {:.4f}".format(epoch_loss, epoch_acc.item()))

#%%
plt.plot(running_loss_history, label="traininig loss")
#%%
plt.plot(running_corrects_history, label="correct history")