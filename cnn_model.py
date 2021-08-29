#%%
from typing import ClassVar
import torch
import torchvision
from imgs_dataset import RHSImgDataset
# import tensorflow as tf
import matplotlib.pyplot as plt
import time
from metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

#%%
class RHS_CNN(torch.nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.MaxPool2d(5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.MaxPool2d(5),
            torch.nn.ReLU(),
            torch.nn.Flatten(),

            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.conv_layers(x)

def train(model, epochs=100):
    writer = SummaryWriter()
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_idx = 64
    losses = []
    for epoch in range(epochs):
        for features, labels in dataloader:
            optimiser.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()
            writer.add_scalar('loss/train', loss.item(), batch_idx)
            batch_idx += 1
        score = f1_score(labels.detach(), output.detach())
        writer.add_scalar('score', score.item(), batch_idx)

# Load n classes and img dataset
n_classes = 50
dataset = RHSImgDataset(n_classes=n_classes)

# Load dataloader
num_workers = 0
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

# Instantiate model
CNN = RHS_CNN(n_classes=n_classes)

# Train model
train(CNN)