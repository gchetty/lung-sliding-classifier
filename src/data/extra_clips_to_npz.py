import cv2
import numpy as np
import pandas as pd
import os
import yaml
from utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']


flow = cfg['PARAMS']['FLOW']

if not (flow == 'Yes'):
    input_folder = cfg['PATHS']['EXTRA_CROPPED_VIDEOS']
    npz_folder = cfg['PATHS']['EXTRA_NPZ']
    refresh_folder(npz_folder)

if not (flow == 'No'):
    flow_input_folder = cfg['PATHS']['EXTRA_FLOW_VIDEOS']
    flow_npz_folder = cfg['PATHS']['EXTRA_FLOW_NPZ']
    refresh_folder(flow_npz_folder)

