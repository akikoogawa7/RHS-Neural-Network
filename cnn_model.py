#%%
from numpy.lib import index_tricks
import torch, torchmetrics
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import ClassVar
from preprocessing.imgs_dataset import RHSImgDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch.utils.tensorboard import SummaryWriter
#%%
class RHSCNN(torch.nn.Module):
    def __init__(self, n_classes, in_channels=3, negative_slope=0.01):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.MaxPool2d(5),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.MaxPool2d(5),
            torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            torch.nn.Dropout(p=0.1),
            torch.nn.Flatten(),

            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            torch.nn.Linear(64, n_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.conv_layers(x)

    def predict(self, x):
        result = self.forward(x)
        return torch.argmax(result, dim=1)

def train(model, epochs=1000, lr = 0.001):
    run_name = f'epochs:{epochs}'
    writer = SummaryWriter(f'runs/{run_name}')
    metric = torchmetrics.Accuracy()
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    batch_idx = 64
    for epoch in range(epochs):
        for train_features, train_labels in train_loader:
            optimiser.zero_grad()
            train_output = model(train_features)
            loss = criterion(train_output, train_labels)
            loss.backward()
            optimiser.step()
            # print(f'Accuracy on batch: {acc}')
            writer.add_scalar('loss/train', loss.item(), batch_idx)
            batch_idx += 1
    for val_features, val_labels in validation_loader:
        val_output = model(val_features)
    train_acc = metric(train_output, train_labels)
    val_acc = metric(val_output, val_labels)
    report(train_acc, val_acc, lr, epoch)
    print(f'Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}')
    # writer.add_scalar('accuracy', acc.item(), batch_idx)
    save_model(epoch, model, optimiser, loss)

def report(train_acc, val_acc, lr, epoch):
    with open('report.txt', 'a') as f:
        f.write(f'\n\nACCURACY SCORE\nTrain accuracy: {train_acc}\nValidation accuracy: {val_acc}\n\nHYPERPARAMETERS\nlr: {lr}\nepochs: {epoch}\n\nTIME\n{time.asctime( time.localtime(time.time()) )}\n\n')

def save_model(epoch, model, optimiser, loss):
    PATH = "state_dict_model.pt"
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser': optimiser.state_dict(),
            'loss': loss,
        },
        PATH,
    )
#%%
def get_train_features_labels():
    for train_features, train_labels in train_loader:
        return train_features, train_labels

def get_val_features_labels():
    for val_features, val_labels in validation_loader:
        return val_features, val_labels
#%%
# Load n classes and img dataset
n_classes = 50
dataset = RHSImgDataset(n_classes=n_classes)

# Creating data indices for training and validation splits
num_workers = 2
validation_split = .2
shuffle_dataset = True
random_seed= 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Create loader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=num_workers, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=num_workers, sampler=valid_sampler)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

# Instantiate model
CNN = RHSCNN(n_classes=n_classes)
#%%
# Train model
train(CNN)

# Load model
def load_model():
    PATH = "state_dict_model.pt"
    CNN.load_state_dict(torch.load(PATH))
    CNN.eval()

#%%
class_names = pd.read_csv('first_50_labels.csv')
class_names
#%%
train_features, train_labels = get_train_features_labels()
val_features, val_labels = get_val_features_labels()

model = RHSCNN(n_classes=n_classes)
y_hat = model.predict(val_features)
y = val_labels.detach()
print(y.shape)
print()
print(y_hat.shape)

#%%
predictions = torch.tensor([[1, 1,],[0, 1]])
labels = predictions
cm = confusion_matrix(y, y_hat)
# plot_confusion_matrix(model, y, y_hat)
plt.show()
# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(20,20))
disp.plot(ax=ax)
plt.show()