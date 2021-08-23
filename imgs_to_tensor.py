#%%
from PIL import Image, UnidentifiedImageError
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
print(len(my_file_list))

#%%

# Convert each img into tensor
def img_to_tensor():
    for file in my_file_list:
        existing_imgs = []
        try:
            img = Image.open(file)
        except:
            continue
        existing_imgs.append(img)
        jpg_to_PIL = torchvision.transforms.ToPILImage()
        PIL_to_tensor = torchvision.transforms.ToTensor()

    plt.imshow(jpg_to_PIL(PIL_to_tensor(img)))

img_to_tensor()
