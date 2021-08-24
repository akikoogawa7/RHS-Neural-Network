#%%
from PIL import Image
import torchvision
import os
import matplotlib.pyplot as plt

# Define file list function
def create_file_list(my_dir, format='.jpg'):
    file_list = []
    print(my_dir)
    for root, dirs, files in os.walk(my_dir, topdown=False):
        for name in files:
            if name.endswith(format):
                full_name = os.path.join(root, name)
                try:
                    Image.open(full_name)
                except:
                    continue
                file_list.append(full_name)
    return file_list

# Load in image folder as file list
my_file_list = create_file_list('./plant_imgs')

