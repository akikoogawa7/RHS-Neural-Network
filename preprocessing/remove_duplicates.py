#%%
import hashlib
from io import IncrementalNewlineDecoder
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline
import time
import numpy as np
#%%
def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()
        
#%%
https://medium.com/@urvisoni/removing-duplicate-images-through-python-23c5fdc7479e 
# %%
