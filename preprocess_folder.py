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
new_dir = 'plant_imgs'
def rename(new_dir):
    for root, dirs, files in os.walk(new_dir, topdown=False):
        for full_label in full_label_list:
            for file in os.listdir(new_dir):
                shutil.move(os.path.join(my_dir), full_label)
                folder = 0
                for folder in range(3297):
                    i = 0
                    for i in range(9):
                        os.rename('/plant_imgs' + f'/{folder}/{i}.jpg', full_label)
                    folder += 1

# %%
rename(new_dir)
# %%
cwd = os.getcwd()
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

# %%
