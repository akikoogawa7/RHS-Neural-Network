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
import time

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

def open_and_scrape_google_imgs(label):
    PATH = 'https://www.google.com/imghp?hl=en'

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.get(PATH)

    # Find search bar
    search_element = driver.find_element_by_name('q')
    search_element.clear()
    driver.implicitly_wait(10)

    # Find I Agree button
    i_agree_button = driver.find_element_by_id('L2AGLb')
    driver.implicitly_wait(10)

    # Click
    ActionChains(driver).move_to_element(i_agree_button).click(i_agree_button).perform()
    ActionChains(driver).move_to_element(search_element).click(search_element).perform()

    # Type in label and enter
    search_element.send_keys(f'{label}')
    search_element.submit()
    driver.implicitly_wait(10)

    # Find image and click
    img = driver.find_element_by_class_name('c7cjWc')
    ActionChains(driver).move_to_element(img).click(img).perform()
    driver.implicitly_wait(10)

    # Will keep scrolling down the webpage until it cannot scroll anymore
    last_height = driver.execute_script('return document.body.scrollHeight')
    while True:
        driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
        time.sleep(2)
        new_height = driver.execute_script('return document.body.scrollHeight')
        try:
            driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
            time.sleep(2)
        except:
            pass
        if new_height == last_height:
            break
        last_height = new_height

    # Take screenshot of 10 imgs
    for i in range(1, 10):
        try:
            driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').screenshot(f'google_imgs_scraped/{label}/{label}('+str(i)+').png')
        except:
            pass
        
for label in label_list:
    open_and_scrape_google_imgs(label)
# %%

# %%
