import yaml
import os
from shutil import rmtree

# Load dictionary of constants stored in config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']

# Obtain input folder paths where the downloaded videos are stored
unmasked_folder = cfg['PATHS']['UNMASKED_VIDEOS']
unmasked_no_sliding_folder = os.path.join(unmasked_folder, 'no_sliding/')

# These input folders need to exist for this to work, so we terminate if they don't
if (not os.path.exists(unmasked_folder)) or (not os.path.exists(unmasked_no_sliding_folder)):
    print('Error: You need to specify a folder for unmasked clips in the config file!')
    exit()

# Ensure that the folder structure exist
masked_folder = cfg['PATHS']['MASKED_VIDEOS']
if not os.path.exists(masked_folder):
    os.makedirs(masked_folder)

masked_no_sliding_folder = os.path.join(masked_folder, 'no_sliding/')
if not os.path.exists(masked_no_sliding_folder):
    os.makedirs(masked_no_sliding_folder)

masking_tool_loc = cfg['PATHS']['MASKING_TOOL']

# Call masking tool
os.system('python auto_masking.py -i='+unmasked_no_sliding_folder + ' -o='+ masked_no_sliding_folder+ ' -f=mp4 -m=' + masking_tool_loc + ' -e=.95 -c=True')

# These folders generated from the masking tool are discarded as they cause complications in to_npz.py
if os.path.exists(os.path.join(masked_no_sliding_folder, 'bad_clips/')):
    rmtree(os.path.join(masked_no_sliding_folder, 'bad_clips/'))
