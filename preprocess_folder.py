#%%
from io import StringIO
import os
import glob
import pandas as pd
import shutil
from urllib.parse import DefragResultBytes, urlparse
from posixpath import dirname

my_dir = './plant_imgs'

# Read in csv
df = pd.read_csv('extracted_links.csv')
df = df.iloc[0:3972]

# %%
# Preprocess to apply named labels to imgs
link_list = []
def create_url_list():
    for link in df['Links']:
        link_list.append(link)
    print(len(link_list))

create_url_list()

#%%
label_list = []
def get_url_label():
    for url in link_list:
        string = url
        parse_object = urlparse(string)
        dir = dirname(parse_object.path)
        segments = dir.rpartition('/')
        string = segments[2]
        label_list.append(string)
        print(string)

get_url_label()

#%%
full_label_list = []
def full_label_filepath(my_dir):
    for root, dirs, files in os.walk(my_dir, topdown=False):
        root = root.rpartition('/')
        root = root[0][1:13]
        for label in label_list:
            full_label_path = os.path.join(root, label)
            full_label_list.append(full_label_path)
        return full_label_list

#%%
full_label_filepath(my_dir)

img_path = os.getcwd() + '/plant_imgs'
print(img_path)

#%%
src_folder_name_list = []
def get_src():
    for root, dirs, files in os.walk(img_path):
        for dir in dirs:
            i = 0
            for i in range(3):
                src_path = img_path + '/' + dir + f'/{i}.jpg'
                i += 1
                src_folder_name_list.append(src_path)
get_src()

src_folder_name_list
 
#%%
def rename():
    for src in src_folder_name_list:
        for label in label_list:
            dst = label
            dst_path = os.path.join(img_path + '/' + dst)
            # for root, dirs, files in os.walk(img_path):
            #     try:
            #         os.rename(root + f'/{i}.jpg', dst_path + f'/{i}.jpg')
            #     except FileNotFoundError:
            #         pass


# %%
rename()

