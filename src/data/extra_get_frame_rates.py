# Saves frame rates of extra (undocumented) clips
# Also deletes clips with frame rates that are not a multiple of 30

import cv2
import numpy as np
import pandas as pd
import os
import yaml
from utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']

input_folder = cfg['PATHS']['EXTRA_CROPPED_VIDEOS']

ids = []
frs = []

for file in os.listdir(input_folder):
    f = os.path.join(input_folder, file)
    cap = cv2.VideoCapture(f)
    fr = round(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fr % 30 == 0:
        ids.append(file[:-4])
        frs.append(fr)

    else:  # delete video
        os.remove(f)

cols = {'id': ids, 'frame_rate': frs}
df = pd.DataFrame(data=cols)

out_path = os.path.join(cfg['PATHS']['CSVS_OUTPUT'], 'extra_clip_frame_rates.csv')
df.to_csv(out_path, index=False)
