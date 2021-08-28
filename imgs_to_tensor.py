#%%
from PIL import Image
import torchvision
import os
import matplotlib.pyplot as plt

# Define file list function
def create_file_list(my_dir, format='.jpg', n_classes=50):
    file_list = []
    for idx, (root, dirs, files) in enumerate(os.walk(my_dir, topdown=False)):
        if idx == n_classes:
            break
        
        for name in files:
            if name.endswith(format):
                full_name = os.path.join(root, name)
                try:
                    Image.open(full_name)
                except:
                    continue
                file_list.append(full_name)

    return file_list

if __name__ == '__main__':
    file_list = create_file_list('plant_imgs', n_classes=50)
    print(len(file_list))
    