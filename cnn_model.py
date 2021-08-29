import torch, torchmetrics
import time
from typing import ClassVar
from imgs_dataset import RHSImgDataset
from torch.utils.tensorboard import SummaryWriter

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

def train(model, epochs=1000):
    writer = SummaryWriter()
    metric = torchmetrics.Accuracy()
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_idx = 64
    for epoch in range(epochs):
        for features, labels in dataloader:
            optimiser.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()
            acc = metric(output, labels)
            # print(f'Accuracy on batch: {acc}')
            writer.add_scalar('loss/train', loss.item(), batch_idx)
            batch_idx += 1
        acc = metric(output, labels)
    report(acc)
    print(f'Accuracy: {acc}')
    # writer.add_scalar('accuracy', acc.item(), batch_idx)
    save_model(epoch, model, optimiser, loss)

def report(scores):
    with open('report.txt', 'w') as f:
        f.write(f'ACCURACY SCORE: {scores} | TIME: {time.asctime( time.localtime(time.time()) )}')
    
def save_model(epoch, model, optimiser, loss):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser': optimiser.state_dict(),
            'loss': loss,
        },
        PATH,
    )

PATH = "state_dict_model.pt"

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

# Load model
def load_model():
    CNN.load_state_dict(torch.load(PATH))
    CNN.eval()
    