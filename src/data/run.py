import os
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']

# Download the videos, then mask them, then convert to miniclips saved in .npz format
# If flow is selected, optical flow frames are extracted between masking and miniclip extraction
# This assumes a Python 3.7 virtual environment has been created and requirements.txt is pip installed

os.system('python download_videos.py')

if cfg['PARAMS']['AMOUNT_ONLY']:
    exit()

os.system('python mask.py')

flow = cfg['PARAMS']['FLOW']

if not (flow == 'No'):
    os.system('python flow.py')

os.system('python to_npz.py')
