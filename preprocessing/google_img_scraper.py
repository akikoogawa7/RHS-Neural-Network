#%%
from selenium import webdriver
from selenium.webdriver.remote import webelement
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os

# Create list from df
def create_label_list_from_df(df):
    label_list = []
    for label in df[0]: label_list.append(label)
    return label_list

labels_df = pd.read_csv('first_80_idx_plant_labels.csv', header=None)
label_list = create_label_list_from_df(labels_df)
label_list
#%%

# Create folders of google imgs
root = os.getcwd()
folder_path = f'{root}/google_imgs_scraped'
os.makedirs(folder_path)

for label in label_list:
    path = os.path.join(folder_path, label)
    os.makedirs(path)

#%%

# def scrape_google_images():
PATH = 'https://www.google.com/imghp?hl=en'

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")

driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get(PATH)


search_element = driver.find_element_by_name('q')
search_element.clear()
driver.implicitly_wait(10)
i_agree_button = driver.find_element_by_id('L2AGLb')
driver.implicitly_wait(10)
ActionChains(driver).move_to_element(i_agree_button).click(i_agree_button).perform()

for label in label_list:
    ActionChains(driver).move_to_element(search_element).click(search_element).perform()
    search_element.send_keys(f'{label} wild')
    break
search_element.submit()


# %%
