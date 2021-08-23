#%%
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from imgs_to_tensor import img_to_tensor, my_file_list

#%%
class RHS_Img_Dataset(torch.utils.data.Dataset):
        def __init__ (self):
            super().__init__
            my_file_list
            img_to_tensor()
            # split data to train and val
            
        def __getitem__(self, index):
            return self.X[index], self.y[index]


        def __len__(self):
            return len(self.y)


# %%
