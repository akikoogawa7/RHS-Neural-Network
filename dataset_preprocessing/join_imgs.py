#%%
import os 
import pandas as pd
import cv2
import numpy as np
from PIL import Image

path = os.getcwd()
# %%
dir_to_google_imgs = f'{path}/google_imgs_scraped'
dir_to_all_imgs = f'{path}/all_plant_imgs'
# %%
google_imgs_folder = 'google_imgs_scraped'
all_plant_imgs_folder = 'all_plant_imgs'

#%%
for google_img in os.listdir(google_imgs_folder):
    all_plant_imgs_folder = f'all_plant_imgs/{google_img}'
    for rhs_img in os.listdir(all_plant_imgs_folder):
        for root, dirs, files in os.walk(all_plant_imgs_folder):
            if google_img == rhs_img:
                new_path = os.path.join('rhs_and_google_imgs', google_img, '/', files)

