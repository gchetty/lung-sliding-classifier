import os
import yaml
import cv2
from utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']


def median_filter(path_in, path_out, kernel_size):

    '''
    Applies median filter to video specified in path_in, and saves to path_out

    :param path_in:
    :param path_out:
    :param kernel_size:
    '''

    cap = cv2.VideoCapture(path_in)
    fr = round(cap.get(cv2.CAP_PROP_FPS))

    frames = []

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            smoothed = cv2.medianBlur(frame, kernel_size)  # Apply median filter
            frames.append(smoothed)

    finally:
        cap.release()

    # Save as video
    size = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fr, size)
    for frame in frames:
        out.write(frame)
    out.release()


crop = cfg['PARAMS']['CROP']
if crop:
    input_folder = cfg['PATHS']['CROPPED_VIDEOS']
else:
    input_folder = cfg['PATHS']['MASKED_VIDEOS']

input_sliding = os.path.join(input_folder, 'sliding/')
input_no_sliding = os.path.join(input_folder, 'no_sliding/')

if not (os.path.exists(input_folder) and os.path.exists(input_sliding) and os.path.exists(input_no_sliding)):
    print('Input directories don\'t exist!')

output_folder = cfg['PATHS']['SMOOTHED_VIDEOS']
output_sliding = os.path.join(output_folder, 'sliding/')
output_no_sliding = os.path.join(output_folder, 'no_sliding/')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_sliding):
    os.makedirs(output_sliding)
if not os.path.exists(output_no_sliding):
    os.makedirs(output_no_sliding)

refresh_folder(output_sliding)
refresh_folder(output_no_sliding)

kernel = cfg['PARAMS']['SMOOTHING_KERNEL_SIZE']

for vid in os.listdir(input_sliding):
    i = os.path.join(input_sliding, vid)
    o = os.path.join(output_sliding, vid)
    median_filter(i, o, kernel)

for vid in os.listdir(input_no_sliding):
    i = os.path.join(input_no_sliding, vid)
    o = os.path.join(output_no_sliding, vid)
    median_filter(i, o, kernel)
