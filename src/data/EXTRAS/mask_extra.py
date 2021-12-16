import yaml
import os
from utils import refresh_folder
from shutil import rmtree

# Load dictionary of constants stored in config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']

# Obtain input folder paths where the downloaded videos are stored
input_folder = cfg['PATHS']['EXTRA_UNMASKED_VIDEOS']

if (not os.path.exists(input_folder)):
    print('Error: You need to specify a folder for unmasked clips in the config file!')
    exit()

output_folder = cfg['PATHS']['EXTRA_MASKED_VIDEOS']
refresh_folder(output_folder)

masking_tool_loc = cfg['PATHS']['MASKING_TOOL']

os.system('python auto_masking.py -i='+input_folder + ' -o='+ output_folder+ ' -f=mp4 -m=' + masking_tool_loc + ' -e=.95 -c=True')

# These folders generated from the masking tool are discarded as they cause complications in to_npz.py
if os.path.exists(os.path.join(output_folder, 'bad_clips/')):
    rmtree(os.path.join(output_folder, 'bad_clips/'))
