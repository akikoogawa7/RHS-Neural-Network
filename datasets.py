#%%
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt

#%%
class RHS_Img_Dataset(torch.utils.data.Dataset):
        def __init__ (self, root, transform):
            super().__init__
            # img to tensor class
            # split data to train and val
            
        def __getitem__(self, index):
            return self.X[index], self.y[index]


        def __len__(self):
            return len(self.y)

