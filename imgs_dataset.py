import torch
import torchvision
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from imgs_to_tensor import create_file_list
from pathlib import Path
from sklearn.model_selection import train_test_split


# Get img properties
my_file_list = create_file_list('./plant_imgs')
im = Image.open(my_file_list[0])
print(f'Image size: {im.size}')
im.show


class RHS_Img_Dataset(torch.utils.data.Dataset):
        def __init__ (self, transform=torchvision.transforms.ToTensor(), target='Full Sun'):
            super().__init__
            seed = 42
            test_size = 0.2

            # Load in categorical variables
            dataset = pd.read_csv('dataset.csv')

            # Load in plant_imgs folder as list 
            self.my_file_list = create_file_list('./plant_imgs')

            # Define target variable
            self.targets = dataset[target]

            # Transform imgs to tensor
            self.transform = transform

            # Split training and validation 10973/2744
            self.X_train, self.X_test, self.y_train, self.y_test  = train_test_split(my_file_list, targets, test_size=test_size, random_state=seed)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(my_file_list, targets, test_size=test_size, random_state=seed)
            
        def __getitem__(self, index):
            # Index through img list
            img_path = self.X_train[index]
            img = Image.open(img_path)

            # Apply tensor transformation to img
            if self.transform:
                img = self.transform(img)
            return img, self.targets[index]

        def __len__(self):
            return len(self.targets)
