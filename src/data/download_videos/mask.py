import yaml
import os
from utils import refresh_folder
from shutil import rmtree

# Load dictionary of constants stored in config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(),"config.yml"), 'r'))

# Obtain input folder paths where the downloaded videos are stored
unmasked_folder = cfg['PATHS']['UNMASKED_VIDEOS']
unmasked_sliding_folder = os.path.join(unmasked_folder, 'sliding/')
unmasked_no_sliding_folder = os.path.join(unmasked_folder, 'no_sliding/')

# These input folders need to exist for this to work, so we terminate if they don't
if (not os.path.exists(unmasked_folder)) or (not os.path.exists(unmasked_sliding_folder)) or (not os.path.exists(unmasked_no_sliding_folder)):
    print('Error: You need to specify a folder for unmasked clips in the config file!')
    exit()

# Ensure all contents in the folder for outputting the masked clips are deleted, but that the folder structure exist
masked_folder = cfg['PATHS']['MASKED_VIDEOS']
refresh_folder(masked_folder)

masked_sliding_folder = os.path.join(masked_folder, 'sliding/')
os.makedirs(masked_sliding_folder)

masked_no_sliding_folder = os.path.join(masked_folder, 'no_sliding/')
os.makedirs(masked_no_sliding_folder)

# First call the masking tool with the sliding parameters, then for no_sliding
os.system('python auto_masking.py -i='+unmasked_sliding_folder + ' -o='+ masked_sliding_folder+ ' -f=mp4 -m=auto_masking_deeper.h5 -e=.95 -c=True')
os.system('python auto_masking.py -i='+unmasked_no_sliding_folder + ' -o='+ masked_no_sliding_folder+ ' -f=mp4 -m=auto_masking_deeper.h5 -e=.95 -c=True')

# These folders generated from the masking tool are discarded as they cause complications in to_npz.py
rmtree(os.path.join(masked_sliding_folder, 'bad_clips/'))
rmtree(os.path.join(masked_no_sliding_folder, 'bad_clips/'))
