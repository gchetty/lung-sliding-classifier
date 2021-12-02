import yaml
import os
from utils import refresh_folder

# Load dictionary of constants stored in config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']

# Obtain input folder paths where the masked videos are stored
cropped_folder = cfg['PATHS']['EXTRA_CROPPED_VIDEOS']

# This input folder needs to exist for this to work, so we terminate if it doesn't
if not os.path.exists(cropped_folder):
    raise NotADirectoryError('You need to specify a folder for masked clips in the config file!')

flow_folder = cfg['PATHS']['EXTRA_FLOW_VIDEOS']
refresh_folder(flow_folder)

# Call optical flow tool
os.system('python denseflow.py --videos_root=' + cropped_folder + ' --data_root=' + flow_folder + ' --new_dir= --num_workers=1')
