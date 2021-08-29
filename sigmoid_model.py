import multiprocessing
from scipy.sparse import data
import torch
import torchvision
import torch.nn.functional as F
import pandas as pd
import multiprocessing
from sklearn import datasets, preprocessing
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class RHS_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        # Read in raw data
        X = pd.read_csv('dataset.csv')
        y = X['Max Time To Ultimate Height']

        # Normalise data
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        # Assert same length in X and y
        assert len(X) == len(y)

        # Transform to tensor
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.y = y.values.reshape(-1, 1)

        # Split data
        X_train, X_test = torch.utils.data.random_split(self.X, [1818, 455])
        self.X_train = X_train

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)
        

class RHS_NeuralNetwork(torch.nn.Module):
    def __init__(self, n_features=8, n_labels=1):
        super().__init__()
        middle_layer = 2 ** 4
        second_middle_layer = 2 ** 5
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, middle_layer),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(middle_layer, second_middle_layer),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(second_middle_layer, n_labels),
            torch.nn.Sigmoid()
        )

    def predict(self, X):
        result = self.forward(X)
        self.rounded_result = result > 0.5

    def forward(self, X):
        return self.layers(X)

def train(model, dataloader, epochs=100):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.0001)
    losses = []
    for epoch in range(epochs):
        # print(next(iter(dataloader)))
        for X, y in dataloader:
            y = y.float()
            y_hat = model(X)
            loss = F.binary_cross_entropy(y_hat, y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            losses.append(loss.item())
            print(loss)
    plt.plot(losses)
    plt.show()


# Instantiate dataset
dataset = RHS_Dataset()

# Instantiate model
model = RHS_NeuralNetwork().float()

# Load in dataloader
num_workers = 0
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

# Train model
train(model, dataloader)