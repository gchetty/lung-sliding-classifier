import cv2
import numpy as np
import os
from utils import refresh_folder
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"config.yml"), 'r'))

def video_to_frames_contig(path, seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE']):
  counter = seq_length
  cap = cv2.VideoCapture(path)
  frames = [[]]
  try:
    while True:
      if counter == 0:
        counter = seq_length
        frames.append([])

      ret, frame = cap.read()
      if not ret:
        break

      frame = cv2.resize(frame, resize)
      frames[-1].append(frame)

      counter -= 1

  finally:
    cap.release()

  # make sure we don't have a dangling (incomplete) sequence at end
  if len(frames[-1]) < seq_length:
    frames = frames[:-1]

  return np.array(frames)

def video_to_npz(path, write_path='', seq_length=50, resize=(128, 128)):
    frames = video_to_frames_contig(path, seq_length, resize)
    np.savez(write_path, frames=frames)

counter = 0

npz_folder = cfg['PATHS']['NPZ']
refresh_folder(npz_folder)

sliding_npz_folder = os.path.join(npz_folder, 'sliding/')
os.makedirs(sliding_npz_folder)

no_sliding_npz_folder = os.path.join(npz_folder, 'no_sliding/')
os.makedirs(no_sliding_npz_folder)

masked_folder = cfg['PATHS']['MASKED_VIDEOS']
masked_sliding_folder = os.path.join(masked_folder, 'sliding/')
masked_no_sliding_folder = os.path.join(masked_folder, 'no_sliding/')

for file in os.listdir(masked_sliding_folder):
  f = os.path.join(masked_sliding_folder, file)
  video_to_npz(f, write_path=(sliding_npz_folder + file[:-4]))
  counter += 1

counter = 0

for file in os.listdir(masked_no_sliding_folder):
  f = os.path.join(masked_no_sliding_folder, file)
  video_to_npz(f, write_path=(no_sliding_npz_folder + file[:-4]))
  counter += 1