import yaml
import os
from utils import refresh_folder
from shutil import rmtree

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"config.yml"), 'r'))

unmasked_folder = cfg['PATHS']['UNMASKED_VIDEOS']
unmasked_sliding_folder = os.path.join(unmasked_folder, 'sliding/')
unmasked_no_sliding_folder = os.path.join(unmasked_folder, 'no_sliding/')

if (not os.path.exists(unmasked_folder)) or (not os.path.exists(unmasked_sliding_folder)) or (not os.path.exists(unmasked_no_sliding_folder)):
    print('Error: You need to specify a folder for unmasked clips in the config file!')
    exit()

masked_folder = cfg['PATHS']['MASKED_VIDEOS']
refresh_folder(masked_folder)

masked_sliding_folder = os.path.join(masked_folder, 'sliding/')
os.makedirs(masked_sliding_folder)

masked_no_sliding_folder = os.path.join(masked_folder, 'no_sliding/')
os.makedirs(masked_no_sliding_folder)

os.system('python auto_masking.py -i='+unmasked_sliding_folder + ' -o='+ masked_sliding_folder+ ' -f=mp4 -m=auto_masking_deeper.h5 -e=.95 -c=True')
os.system('python auto_masking.py -i='+unmasked_no_sliding_folder + ' -o='+ masked_no_sliding_folder+ ' -f=mp4 -m=auto_masking_deeper.h5 -e=.95 -c=True')

rmtree(os.path.join(masked_sliding_folder, 'bad_clips/'))
rmtree(os.path.join(masked_no_sliding_folder, 'bad_clips/'))