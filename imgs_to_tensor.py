#%%
from PIL import Image
import torchvision
import os
import matplotlib.pyplot as plt

# %%
# transform should be in class

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

#%%

# # Convert each img into tensor
# def img_to_tensor():
#     for file in my_file_list:
#         existing_imgs = []
#         try:
#             img = Image.open(file)
#         except:
#             continue
#         existing_imgs.append(img)
#         jpg_to_PIL = torchvision.transforms.ToPILImage()
#         PIL_to_tensor = torchvision.transforms.ToTensor()

#     plt.imshow(jpg_to_PIL(PIL_to_tensor(img)))

# img_to_tensor()

# %%
