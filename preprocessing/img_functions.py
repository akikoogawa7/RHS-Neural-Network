#%%
import os
import pandas as pd
from pandas import DataFrame
from PIL import Image
from string import ascii_lowercase, ascii_uppercase

# Define file list function
def first_letter_alphabetical_order(name):
    for letter in ascii_uppercase:
        if name.startswith(letter):
            return name
        else:
            pass

def create_file_list(my_dir, format='.jpg', n_classes=50):
    file_list = []
    for idx, (root, dirs, files) in enumerate(os.walk(my_dir, topdown=False)):
        if idx == n_classes:
            break
        
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

#%%
file_list = create_file_list('dataset/plant_imgs', n_classes=50)
#%%
import pandas as pd
#%%
file_list
#%%
extract_species_list = []

def extract_species(img_path):
    species = img_path.split('/')
    species = species[-2]
    return species

for file in file_list:
    extracted_species = extract_species(file)
    extract_species_list.append(extracted_species)
    
extract_species_list

#%%
import pandas as pd
from pandas import DataFrame

extracted_species_df = DataFrame(extract_species_list, columns=['Species'])
species_link_df = DataFrame(file_list, columns=['Path'])

# extracted species dataframe
extracted_species_df

# species img path dataframe
species_link_df

extracted_species_df = extracted_species_df.drop_duplicates(subset='Species', keep='first')
#%%
extracted_species_df.to_csv('first_52_plants.csv')

#%%
combined_df = [extracted_species_df, species_link_df]
concatenated_dfs = pd.concat(combined_df, axis=1)
concatenated_dfs

#%%
if __name__ == '__main__':
    file_list = create_file_list('plant_imgs', n_classes=50)
    print(len(file_list))   
# %%
