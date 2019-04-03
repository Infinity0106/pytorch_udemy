#%%
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import PIL
from PIL import Image
!pip install torch torchvision

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

#%%
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
training_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
validation_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(
    dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset, batch_size=100, shuffle=False)

#%%


def image_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image


#%%
dataiter = iter(training_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(image_convert(images[idx]))
  ax.set_title([labels[idx].item()])


#%%
class LeNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4*4*50, 500)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)

    # reshape to fully connected
    x = x.view(-1, 4*4*50)
    x = F.relu(self.fc1(x))
    x = self.dropout1(x)
    x = self.fc2(x)

    return x


#%%
model = LeNet().to(device)
model

#%%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#%%
epochs = 15
running_loss_history = []
running_corrects_history = []
validation_run_loss_history = []
validation_run_corrects_history = []

for e in range(epochs):
  running_loss = 0.0
  running_corrects = 0.0
  validation_running_loss = 0.0
  validation_running_corrects = 0.0

  for inputs, labels in training_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs, 1)
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)

  else:
    with torch.no_grad():
      for val_input, val_labels in validation_loader:
        val_input = val_input.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_input)
        val_loss = criterion(val_outputs, val_labels)

        _, val_preds = torch.max(val_outputs, 1)
        validation_running_loss += val_loss.item()
        validation_running_corrects += torch.sum(val_preds == val_labels.data)

    epoch_loss = running_loss/len(training_loader)
    epoch_acc = running_corrects.float()/len(training_loader)
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc.item())

    validation_epoch_loss = validation_running_loss/len(validation_loader)
    validation_epoch_acc = validation_running_corrects.float()/len(validation_loader)
    validation_run_loss_history.append(validation_epoch_loss)
    validation_run_corrects_history.append(validation_epoch_acc.item())
    print("epoch {}".format(e+1))
    print("Training loss: {:.4f}, {:.4f}".format(epoch_loss, epoch_acc.item()))
    print("Validation loss: {:.4f}, {:.4f}".format(
        validation_epoch_loss, validation_epoch_acc.item()))

#%%
plt.plot(running_loss_history, label="traininig loss")
plt.plot(validation_run_loss_history, label="validation loss")
plt.legend()
#%%
plt.plot(running_corrects_history, label="correct history")
plt.plot(validation_run_corrects_history, label="validation correct")
plt.legend()

#%%
plt.plot(running_loss_history, label="Running loss")
plt.plot(validation_run_loss_history, label="Validation loss")
plt.legend()


#%%
# final test real image
# img = cv2.imread("5.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = Image.open("5.jpg")
img = PIL.ImageOps.invert(img)
img = img.convert('1')
img = transform(img)
plt.imshow(image_convert(img))


#%%
img = img.to(device)
img = img[0].unsqueeze(0).unsqueeze(0)
output = model(img)

_, pred = torch.max(output, 1)
print("Prediction {}".format(pred.item()))

#%%
dataiter = iter(training_loader)
images, labels = dataiter.next()
images_ = images.to(device)
output = model(images_)

_, pred = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(image_convert(images[idx]))
  ax.set_title("{} {}".format(str(pred[idx].item()), str(
      labels[idx].item())), color=("green" if pred[idx] == labels[idx] else "red"))
