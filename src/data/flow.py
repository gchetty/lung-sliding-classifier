import yaml
import os
from utils import refresh_folder

# Load dictionary of constants stored in config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']
crop = cfg['PARAMS']['CROP']
smooth = cfg['PARAMS']['SMOOTHING']

# Obtain input folder paths where the masked videos are stored
if smooth:
    input_folder = cfg['PATHS']['SMOOTHED_VIDEOS']
elif crop:
    input_folder = cfg['PATHS']['CROPPED_VIDEOS']
else:
    input_folder = cfg['PATHS']['MASKED_VIDEOS']
input_sliding_folder = os.path.join(input_folder, 'sliding/')
input_no_sliding_folder = os.path.join(input_folder, 'no_sliding/')

# These input folders need to exist for this to work, so we terminate if they don't
if (not os.path.exists(input_folder)) or (not os.path.exists(input_sliding_folder)) or (not os.path.exists(input_no_sliding_folder)):
    raise NotADirectoryError('You need to specify a folder for masked / cropped clips in the config file!')

# Ensure all contents in the folder for outputting the flow clips are deleted, but that the folder structure exist
flow_folder = cfg['PATHS']['FLOW_VIDEOS']
refresh_folder(flow_folder)

flow_sliding_folder = os.path.join(flow_folder, 'sliding/')
os.makedirs(flow_sliding_folder)

flow_no_sliding_folder = os.path.join(flow_folder, 'no_sliding/')
os.makedirs(flow_no_sliding_folder)

# Call optical flow tool
os.system('python denseflow.py --videos_root=' + input_folder + ' --data_root=' + flow_folder + ' --crop=' + str(crop))
