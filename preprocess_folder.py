#%%
from io import StringIO
import os
import glob
import pandas as pd
import shutil
from urllib.parse import urlparse
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

get_url_label()

#%%
full_label_list = []
def full_label_filepath(my_dir):
    for root, dirs, files in os.walk(my_dir, topdown=False):
        root = root.rpartition('/')
        root = root[0]
        for label in label_list:
            full_label_path = os.path.join(root, label)
            full_label_list.append(full_label_path)
        return full_label_list

#%%
full_label_filepath(my_dir)

#%%
full_path = os.getcwd()
new_dir = full_path + '/plant_imgs/'
print(new_dir)

def rename(new_dir):
    for root, dirs, files in os.walk(new_dir):
        for full_label in full_label_list:
            full_label = full_label[13::]
            folder = 0
            for folder in range(3297):
                if not os.path.exists(new_dir):
                    os.makedirs(os.path.dirname(root), exist_ok=True)
                i = 0
                for i in range(9):
                    os.rename(root + f'{folder}/{i}.jpg', new_dir + full_label)
                folder += 1

# %%
rename(new_dir)


