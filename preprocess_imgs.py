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
df = pd.read_csv('extracted_links.csv', header=None)
df = df.iloc[0:3972]
#%%
display(df.head(15))
#%%
df.iloc[41].head(1).item()
# %%

# Preprocess to apply named labels to imgs
link_list = []
def create_url_list():
    for link in df[0]:
        link_list.append(link)
    print(len(link_list))

create_url_list()

#%%
label_list = []
def get_url_label(url):
    string = url
    parse_object = urlparse(string)
    dir = dirname(parse_object.path)
    segments = dir.rpartition('/')
    string = segments[2]
    label_list.append(string)
    return(string)

#%%

# Create column with labels
df['Labels'] = df.apply(lambda row: get_url_label(row[0]), axis=1)

#%%

# Rename files with labels
img_path = os.getcwd() + '/plant_imgs'
print(img_path)

for index, row in df.iterrows():
    print()
    print(index +1)
    print(row['Labels'])
    dst_path = os.path.join(img_path, row['Labels'])
    src_path = os.path.join(img_path, str(index +1))
    print(dst_path)
    print()
    print(src_path)
    try:
        os.rename(src_path, dst_path)
    except FileNotFoundError as E:
        print(E)
    try:
        os.rename(src_path, dst_path)
    except OSError as E:
        print(E)
