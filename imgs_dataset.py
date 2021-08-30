import torch
import torchvision
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from imgs_to_tensor import create_file_list
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import train_test_split

name_to_index = {
    name: idx for idx, name in enumerate(os.listdir('plant_imgs'))
}

default_transform = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.CenterCrop(200),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

class RHSImgDataset(torch.utils.data.Dataset):
        def __init__ (self, transform=default_transform, target='Full Sun', n_classes=50):
            super().__init__

            # Load in plant_imgs folder as list 
            self.my_file_list = create_file_list('./plant_imgs', n_classes=n_classes)

            # Transform imgs to tensor
            self.transform = transform

        def __getitem__(self, index):
            # Index through img list
            img_path = self.my_file_list[index]
            species = img_path.split('/')
            species = species[-2]
            idx = name_to_index[species]
            img = Image.open(img_path)

            # Apply tensor transformation to img
            if self.transform:
                img = self.transform(img)
            return img, idx
            

        def __len__(self):
            return len(self.my_file_list)

if __name__ == '__main__':
    dataset = RHSImgDataset()
    dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    for data in dataloader:
        X, y = data
        print(f'The size of X: {X.shape}, the size of y: {y.shape}')
        print(f'The dimension of X: {X.ndim}, the dimension of y: {y.ndim}')
        break