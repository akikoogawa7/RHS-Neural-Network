import os
from numpy import concatenate
import pandas as pd
from pandas import DataFrame
from PIL import Image
from string import ascii_lowercase, ascii_uppercase

# Define file list functions for image folder processing into computer readables list of files

# Function to order files in alphabetical order
def first_letter_alphabetical_order(name):
    for letter in ascii_uppercase:
        if name.startswith(letter):
            return name
        else:
            pass

# Function to create file list from plant_imgs folder
# For RHS imgs
def create_file_list(my_dir, format='.jpg'):
    file_list = []
    for root, (dirs, files) in enumerate(os.walk(my_dir, topdown=False)):

        for name in files:
            first_letter_alphabetical_order(name)
            if name.endswith(format):
                full_name = os.path.join(root, name)
                try:
                    Image.open(full_name)
                except:
                    continue
                file_list.append(full_name)
    return file_list

# For Google imgs
def create_file_list_google_imgs(my_dir, format='.png'):
    file_list = []
    for root, dirs, files in os.walk(my_dir, topdown=False):      
        for name in files:
            if name.endswith(format):
                full_name = os.path.join(root, name)
                file_list.append(full_name)
    return file_list

# Function to append img path strings in df column - 'Path' to a list
def create_img_paths_list_from_df(df):
    img_path_list = []
    for img_path in df: img_path_list.append(img_path)
    return img_path_list

# Function to split the string to get 'Species' label
def extract_species(img_path):
    species = img_path.split('/')
    species = species[-2]
    return species

# Function to convert df to csv files
def df_to_csv(df, csv_filename):
    df.to_csv(f'{csv_filename}.csv')

# Function to convert df to csv with no column header
def df_to_csv_no_header(df, csv_filename):
    df.to_csv(f'{csv_filename}.csv', header=False, index=False)

# Create python-readable file list from plant_imgs folder
file_list = create_file_list('all_plant_imgs', n_classes=80)
file_list

# Get each 'Species' label from file path
extract_species_list = []
for file in file_list:
    extracted_species = extract_species(file)
    extract_species_list.append(extracted_species)

# Turn each list into df columns
# Drop duplicates from 'Species' column ready for inner join with 'Path' column
extracted_species_df = DataFrame(extract_species_list, columns=['Species']).drop_duplicates(subset='Species', keep='first')
species_img_path_df = DataFrame(file_list, columns=['Path'])

# Sort to alphabetical order
extracted_species_df = extracted_species_df.sort_values('Species')

# Save extracted species label df with dropped duplicates to csv
df_to_csv_no_header(extracted_species_df, 'first_80_idx_plant_labels')

# Define combination of dfs to concatenate
combined_df = [extracted_species_df, species_img_path_df]

# Concatenate dfs using inner join to drop duplicate values in 'Species' and in 'Path'
concatenated_dfs = pd.concat(combined_df, join='inner', axis=1)
df_to_csv(concatenated_dfs, 'combined_plant_labels_and_path')

# Create img paths list from concatenated df
img_path_file_list = create_img_paths_list_from_df(concatenated_dfs)

# Drop column from the concatenated df to only keep paths
paths_df = concatenated_dfs.drop(columns='Species', axis=1)
df_to_csv_no_header(paths_df, 'first_80_idx_plant_paths')

# 'Paths' column with img paths now no longer have duplicates 

if __name__ == '__main__':
    file_list = create_file_list('plant_imgs', n_classes=80)
    print(len(file_list))