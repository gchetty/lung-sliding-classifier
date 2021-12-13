import os
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']['PARAMS']

# Download the videos, then mask them, then convert to miniclips saved in .npz format
# If flow is selected, optical flow frames are extracted between masking and miniclip extraction
# This assumes a Python 3.7 virtual environment has been created and requirements.txt is pip installed

flow = cfg['FLOW']
crop = cfg['CROP']
smooth = cfg['SMOOTHING']
amount_only = cfg['AMOUNT_ONLY']

os.system('python download_videos.py')

if amount_only:
    exit()

os.system('python mask.py')

if crop:
    print()  # CALL CROPPING SCRIPT

# Smoothing
if smooth:
    os.system('python smooth.py')

if not (flow == 'No'):
    os.system('python flow.py')

# Convert masked/cropped/flow videos to npz
if flow == 'Yes':
    os.system('python to_npz.py --flow=True --crop=' + str(crop))
elif flow == 'No':
    os.system('python to_npz.py --flow=False --crop=' + str(crop))
else:
    os.system('python to_npz.py --flow=True --crop=' + str(crop))
    os.system('python to_npz.py --flow=False --crop=' + str(crop))
