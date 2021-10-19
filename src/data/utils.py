import os
from shutil import rmtree


def refresh_folder(path):
    '''
    Ensures the folder exists and is empty
    :param path: Path to the folder of interest
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        rmtree(path)
        os.makedirs(path)