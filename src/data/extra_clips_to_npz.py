import cv2
import numpy as np
import pandas as pd
import os
import yaml
from utils import refresh_folder
from .to_npz import video_to_npz, flow_frames_to_npz_contig, flow_frames_to_npz_downsampled

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']

flow = cfg['PARAMS']['FLOW']

if not (flow == 'Yes'):
    input_folder = cfg['PATHS']['EXTRA_CROPPED_VIDEOS']
    npz_folder = cfg['PATHS']['EXTRA_NPZ']
    refresh_folder(npz_folder)

if not (flow == 'No'):
    flow_input_folder = cfg['PATHS']['EXTRA_FLOW_VIDEOS']
    flow_npz_folder = cfg['PATHS']['EXTRA_FLOW_NPZ']
    refresh_folder(flow_npz_folder)

csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']

# Read frame rate csv
fr_path = os.path.join(csv_out_folder, 'extra_clip_frame_rates.csv')
fr_df = pd.read_csv(fr_path)

# Each element is (mini-clip_id) for download as csv
df_rows = []
df_rows_flow = []

if not (flow == 'No'):

    for id in os.listdir(flow_input_folder):
        path = os.path.join(flow_input_folder, id)
        fr = ((fr_df[fr_df['id'] == id])['frame_rate']).values[0]
        if fr == 30:
            flow_frames_to_npz_contig(path, orig_id=id, patient_id='N/A', df_rows=df_rows_flow,
                                      write_path=(flow_npz_folder + id))
        else:
            flow_frames_to_npz_downsampled(path, orig_id=id, patient_id='N/A', df_rows=df_rows_flow,
                                           write_path=(flow_npz_folder + id))

if not (flow == 'Yes'):

    for file in os.listdir(input_folder):
        f = os.path.join(input_folder, file)
        video_to_npz(f, orig_id=file[:-4], patient_id='N/A', df_rows=df_rows, write_path=(npz_folder + file[:-4]))

# Download dataframes linking mini-clip ids and patient ids as csv files
if df_rows:
    csv_out_path = os.path.join(csv_out_folder, 'extra_mini_clips.csv')
    out_df = pd.Dataframe(df_rows, columns=['id', 'patient_id'])
    out_df.to_csv(csv_out_path, index=False)
if df_rows_flow:
    csv_out_path_flow = os.path.join(csv_out_folder, 'extra_flow_mini_clips.csv')
    out_df_flow = pd.Dataframe(df_rows_flow, columns=['id', 'patient_id'])
    out_df_flow.to_csv(csv_out_path_flow, index=False)
