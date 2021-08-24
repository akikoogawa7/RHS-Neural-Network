#%%
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from imgs_to_tensor import my_file_list
import pandas as pd

#%%
class RHS_Img_Dataset(torch.utils.data.Dataset):
        def __init__ (self, transform=torchvision.transforms.ToTensor(), target='Full Sun'):
            super().__init__
            dataset = pd.read_csv('dataset.csv')
            self.targets = dataset[target]
            self.transform = transform
            my_file_list
            # split data to train and val
            
        def __getitem__(self, index):
            img_path = my_file_list[index]
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            return img, self.targets[index]

        def __len__(self):
            return len(self.y)


# %%
