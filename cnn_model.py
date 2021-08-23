#%%
import torch
import torchvision
from imgs_dataset import RHS_Img_Dataset
from torch.utils.tensorboard import SummaryWriter

#%%
# Load in img dataset
rhs_imgs = torch.utils.data.Dataset.RHS_Img_Dataset()

class RHS_CNN(torch.nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2D(in_channels, 32, kernal_size=3),
            torch.nn.MaxPool2D(2),
            torch.nn.BatchNorm2D(32), # do you batch norm once?
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv2D(32, 64, kernal_size=3),
            torch.nn.MaxPool2D(2),
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
        for features, labels in rhs_imgs:
            optimiser.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()
            writer.add_scalar('loss/train', loss.item(), batch_idx)
            batch_idx += 1

cnn = RHS_CNN()
train(cnn)
