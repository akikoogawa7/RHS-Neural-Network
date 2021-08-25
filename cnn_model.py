#%%
from typing import ClassVar
import torch
import torchvision
from imgs_dataset import RHS_Img_Dataset
from torch.utils.tensorboard import SummaryWriter

#%%
# Load in img dataset
rhs_img_dataset = RHS_Img_Dataset('Full Sun')

#%%
class RHS_CNN(torch.nn.Module):
    def __init__(self, in_channels, n_classes):
        super(RHS_CNN, self).__init__
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=3),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32), # do you batch norm once?
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),

            # torch.nn.Linear(how many input features after conv?, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, how many k outputs depends on what I classify),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        self.conv_layers(x)

def train(model, epochs=100):
    writer = SummaryWriter()
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    batch_idx = 64
    losses = []
    for epoch in range(epochs):
        for features, labels in rhs_imgs_dataset:
            optimiser.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()
            writer.add_scalar('loss/train', loss.item(), batch_idx)
            batch_idx += 1
#%%
model = RHS_CNN(3, 1)
# train(cnn)

# %%

# %%
