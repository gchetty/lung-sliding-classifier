import os
from shutil import rmtree

def refresh_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        rmtree(path)
        os.makedirs(path)