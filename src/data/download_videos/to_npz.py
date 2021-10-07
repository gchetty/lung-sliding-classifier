import cv2
import numpy as np
import pandas as pd
import os
from utils import refresh_folder
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"config.yml"), 'r'))

def video_to_frames_strided(path, orig_id, patient_id, df_rows, stride=cfg['PARAMS']['STRIDE'], seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):

  # IF YOU CAN DETERMINE # FRAMES IN VIDEO FIRST, WE CAN CHECK VALIDITY CONDITION HERE

  cap = cv2.VideoCapture(path)
  frames = []
  for i in range(stride):
    frames.append([])

  index = 0

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = cv2.resize(frame, tuple(resize))
      frames[index].append(frame)
      index = (index + 1) % stride

  finally:
    cap.release()

  # Clip is invalid if necessary sequence length is longer than mini-clips created with given stride length
  # Last index guaranteed to have sequence of equal length or shorter by 1
  if len(frames[-1]) < seq_length:
    return

  # Split into mini-clips with given sequence length, update df rows, and write npz files
  counter = 1
  for set in frames:  # iterate through each 'set' of frames (which make up 1+ mini-clips)
    num_mini_clips = int(len(set) / seq_length)  # rounded down
    for i in range(num_mini_clips):
      df_rows.append([orig_id + '_' + str(counter), patient_id])
      np.savez(write_path + '_' + str(counter), frames=set[i*seq_length:i*seq_length+seq_length])
      counter += 1

  return

def video_to_frames_contig(path, orig_id, patient_id, df_rows, seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):
  counter = seq_length
  mini_clip_num = 1
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      if counter == 0:
        df_rows.append([orig_id + '_' + str(mini_clip_num), patient_id])  # append to what will make output dataframes
        np.savez(write_path + '_' + str(mini_clip_num), frames=frames)  # output
        counter = seq_length
        mini_clip_num += 1
        frames = []

      ret, frame = cap.read()
      if not ret:
        break

      frame = cv2.resize(frame, tuple(resize))
      frames.append(frame)

      counter -= 1

  finally:
    cap.release()

  return

def video_to_npz(path, orig_id, patient_id, df_rows, write_path='', method=cfg['PARAMS']['METHOD']):
  if method == 'Contiguous':
    video_to_frames_contig(path, orig_id, patient_id, df_rows, write_path=write_path)
  else:
    video_to_frames_strided(path, orig_id, patient_id, df_rows, write_path=write_path)


npz_folder = cfg['PATHS']['NPZ']
refresh_folder(npz_folder)

sliding_npz_folder = os.path.join(npz_folder, 'sliding/')
os.makedirs(sliding_npz_folder)

no_sliding_npz_folder = os.path.join(npz_folder, 'no_sliding/')
os.makedirs(no_sliding_npz_folder)

masked_folder = cfg['PATHS']['MASKED_VIDEOS']
masked_sliding_folder = os.path.join(masked_folder, 'sliding/')
masked_no_sliding_folder = os.path.join(masked_folder, 'no_sliding/')

df_rows_sliding = []
df_rows_no_sliding = []

csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']
sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'sliding.csv'))
no_sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding.csv'))

for file in os.listdir(masked_sliding_folder):
  f = os.path.join(masked_sliding_folder, file)
  patient_id = ((sliding_df[sliding_df['id'] == file[:-4]])['patient_id']).values[0]
  video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_sliding, write_path=(sliding_npz_folder + file[:-4]))

for file in os.listdir(masked_no_sliding_folder):
  f = os.path.join(masked_no_sliding_folder, file)
  patient_id = ((no_sliding_df[no_sliding_df['id'] == file[:-4]])['patient_id']).values[0]
  video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_no_sliding, write_path=(no_sliding_npz_folder + file[:-4]))

out_df_sliding = pd.DataFrame(df_rows_sliding, columns=['id', 'patient_id'])
csv_out_path_sliding = os.path.join(csv_out_folder, 'sliding_mini_clips.csv')
out_df_sliding.to_csv(csv_out_path_sliding, index=False)

out_df_no_sliding = pd.DataFrame(df_rows_no_sliding, columns=['id', 'patient_id'])
csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_mini_clips.csv')
out_df_no_sliding.to_csv(csv_out_path_no_sliding, index=False)
