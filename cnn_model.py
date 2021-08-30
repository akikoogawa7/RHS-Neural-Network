from numpy.lib import index_tricks
import torch, torchmetrics
import time
import numpy as np
from typing import ClassVar
from imgs_dataset import RHSImgDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

class RHSCNN(torch.nn.Module):
    def __init__(self, n_classes, in_channels=3, negative_slope=0.01):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.MaxPool2d(5),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.MaxPool2d(5),
            torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            torch.nn.Dropout(p=0.2),
            torch.nn.Flatten(),

            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            torch.nn.Linear(64, n_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.conv_layers(x)

def train(model, epochs=1000, lr = 0.001):
    writer = SummaryWriter()
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
        f.write(f'ACCURACY SCORE\nTrain accuracy: {train_acc}\nValidation accuracy: {val_acc}\n\nHYPERPARAMETERS\nlr: {lr}\nepochs: {epoch}\n\nTIME\n{time.asctime( time.localtime(time.time()) )}\n\n')
    
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

# Load n classes and img dataset
n_classes = 50
dataset = RHSImgDataset(n_classes=n_classes)

# Creating data indices for training and validation splits
num_workers = 0
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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# Instantiate model
CNN = RHSCNN(n_classes=n_classes)

# Train model
train(CNN)

# Load model
def load_model():
    PATH = "state_dict_model.pt"
    CNN.load_state_dict(torch.load(PATH))
    CNN.eval()