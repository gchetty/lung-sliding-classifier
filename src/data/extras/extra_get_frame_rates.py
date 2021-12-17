# Saves frame rates of extra (undocumented) clips

import cv2
import pandas as pd
import os
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']

input_folder = cfg['PATHS']['EXTRA_UNMASKED_VIDEOS']

ids = []
frs = []

for file in os.listdir(input_folder):
    f = os.path.join(input_folder, file)
    cap = cv2.VideoCapture(f)
    fr = round(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    ids.append(file[:-4])
    frs.append(fr)

cols = {'id': ids, 'frame_rate': frs}
df = pd.DataFrame(data=cols)

out_path = os.path.join(cfg['PATHS']['CSVS_OUTPUT'], 'extra_clip_frame_rates.csv')
df.to_csv(out_path, index=False)
