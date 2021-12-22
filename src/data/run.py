import os
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']['PARAMS']

# Download the videos, then mask them, then convert to miniclips saved in .npz format
# If flow is selected, optical flow frames are extracted between masking and miniclip extraction
# This assumes that requirements.txt is pip installed

# TO PREPARE FOR M-MODE:
#    Ensure FLOW = 'No'
#    Ensure M_MODE = True

flow = cfg['FLOW']
smooth = cfg['SMOOTHING']
m_mode = cfg['M_MODE']
amount_only = cfg['AMOUNT_ONLY']

if m_mode:
    if not (flow == 'No'):
        raise AssertionError('Please set flow to \'No\' in the config file!')

os.system('python download_videos.py')

if amount_only:
    exit()

os.system('python mask.py')  # accommodate for new masking tool ?

# Smoothing
if smooth:
    os.system('python smoothing_filter.py')  # FIX THIS AS PART OF PR

if not (flow == 'No'):
    os.system('python flow.py')

# Convert masked/flow videos to npz
if flow == 'Yes':
    os.system('python to_npz.py --flow=True --smooth=' + str(smooth))
elif flow == 'No':
    os.system('python to_npz.py --smooth=' + str(smooth))
else:
    os.system('python to_npz.py --flow=True --smooth=' + str(smooth))
    os.system('python to_npz.py --smooth=' + str(smooth))
