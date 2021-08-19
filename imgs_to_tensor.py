#%%
from PIL import Image
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt

# %%
# Define file list function
def create_file_list(my_dir, format='.jpg'):
    file_list = []
    print(my_dir)
    for root, dirs, files in os.walk(my_dir, topdown=False):
        for name in files:
            if name.endswith(format):
                full_name = os.path.join(root, name)
                file_list.append(full_name)
    return file_list

# Load in image folder as file list
my_file_list = create_file_list('./plant_imgs')

#%%
# Convert each file into tensor
for file in my_file_list:
    img = Image.open(file)
    jpg_to_PIL = torchvision.ToPILImage()
    img_to_tensor = jpg_to_PIL.ToTensor()
    plt.imshow(jpg_to_PIL(img_to_tensor(img)))
