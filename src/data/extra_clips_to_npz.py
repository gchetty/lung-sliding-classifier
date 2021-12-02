import cv2
import numpy as np
import pandas as pd
import os
import yaml
from utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']


def video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, seq_length=cfg['PARAMS']['WINDOW'],
                                resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):
    '''
    Converts a LUS video file to mini-clips downsampled to 30 FPS with specified sequence length

    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param cap: Captured video of full clip, returned by cv2.VideoCapture()
    :param fr: Frame rate (integer) of original clip - MUST be divisible by 30
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    '''

    # Check validity of frame rate param
    assert (isinstance(fr, int))
    assert (fr % 30 == 0)

    frames = []
    stride = fr // 30

    index = 0  # Position in 'frames' array where the next frame is to be appended

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index == 0:  # Add every nth frame only
                frame = cv2.resize(frame, tuple(resize))
                frames.append(frame)
            index = (index + 1) % stride

    finally:
        cap.release()

    # Assemble and save mini-clips from extracted frames
    counter = 1
    num_mini_clips = len(frames) // seq_length
    for i in range(num_mini_clips):
        df_rows.append([orig_id + '_' + str(counter), patient_id])
        np.savez(write_path + '_' + str(counter), frames=frames[i * seq_length:i * seq_length + seq_length])
        counter += 1

    return


def video_to_frames_contig(orig_id, patient_id, df_rows, cap, seq_length=cfg['PARAMS']['WINDOW'],
                           resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):
    '''
    Converts a LUS video file to contiguous-frame mini-clips with specified sequence length

    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param cap: Captured video of full clip, returned by cv2.VideoCapture()
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    '''

    counter = seq_length
    mini_clip_num = 1  # nth mini-clip being made from the main clip
    frames = []
    try:
        while True:

            # When seq_length frames have been read, update df rows and write npz files
            # The id of the xth mini-clip from a main clip is the id of the main clip with _x appended to it
            if counter == 0:
                df_rows.append(
                    [orig_id + '_' + str(mini_clip_num), patient_id])  # append to what will make output dataframes
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


def flow_frames_to_npz_downsampled(path, orig_id, patient_id, df_rows, fr, seq_length=cfg['PARAMS']['WINDOW'],
                                   resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):
    '''
    Converts a directory of x and y flow frames to mini-clips downsampled to 30 FPS, with flows stacked on axis = -1

    :param path: Path to directory containing flow frames
    :param orig_id: ID of the LUS video file associated with the flow frames
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param fr: Frame rate (in FPS) of original LUS clip
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    '''

    frames_x = []
    frames_y = []

    stride = fr // 30

    # Read all flow frames
    for file in os.listdir(path):
        frame = cv2.imread(os.path.join(path, file), 0)
        frame = cv2.resize(frame, tuple(resize))
        ind = int(file[7:12])  # flow frame number
        if (ind - 1) % stride == 0:  # take every nth flow frame only
            if '_x_' in file:
                frames_x.append(frame)
            else:
                frames_y.append(frame)

    counter = 1
    num_mini_clips = len(frames_x) // seq_length

    # Stack x and y flow (making 2 channels) and save mini-clip sequences
    for i in range(num_mini_clips):
        df_rows.append([orig_id + '_' + str(counter), patient_id])
        x_seq = np.array(frames_x[i * seq_length:i * seq_length + seq_length])
        y_seq = np.array(frames_y[i * seq_length:i * seq_length + seq_length])
        np.savez(write_path + '_' + str(counter), frames=np.stack((x_seq, y_seq), axis=-1))
        counter += 1

    return


def flow_frames_to_npz_contig(path, orig_id, patient_id, df_rows, seq_length=cfg['PARAMS']['WINDOW'],
                              resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):
    '''
    Converts a directory of x and y flow frames to contiguous mini-clips, with flows stacked on axis = -1

    :param path: Path to directory containing flow frames
    :param orig_id: ID of the LUS video file associated with the flow frames
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    '''

    frames_x = []
    frames_y = []

    # Read all flow frames
    for file in os.listdir(path):
        frame = cv2.imread(os.path.join(path, file), 0)
        frame = cv2.resize(frame, tuple(resize))
        if '_x_' in file:
            frames_x.append(frame)
        else:
            frames_y.append(frame)

    counter = 1
    num_mini_clips = len(frames_x) // seq_length

    # Stack x and y flow (making 2 channels) and save mini-clip sequences
    for i in range(num_mini_clips):
        df_rows.append([orig_id + '_' + str(counter), patient_id])
        x_seq = np.array(frames_x[i * seq_length:i * seq_length + seq_length])
        y_seq = np.array(frames_y[i * seq_length:i * seq_length + seq_length])
        np.savez(write_path + '_' + str(counter), frames=np.stack((x_seq, y_seq), axis=-1))
        counter += 1

    return


def video_to_npz(path, orig_id, patient_id, df_rows, write_path='', method=cfg['PARAMS']['METHOD'], fr=None):
    '''
    Converts a LUS video file to mini-clips

    :param path: Path to video file to be converted
    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param write_path: Path to directory where output mini-clips are saved
    :param method: Method of frame extraction for mini-clips, either 'Contiguous' or ' Stride'
    '''

    flow = cfg['PARAMS']['FLOW']

    cap = cv2.VideoCapture(path)

    if not fr:
        fr = round(cap.get(cv2.CAP_PROP_FPS)) \
            # Disregard clips with undesired frame rate, only if frame rate not passed in (passed in = override checking)
        if not (fr % 30 == 0):
            return
    else:  # cast frame rate to closest multiple of 30
        fr = round(fr / 30.0) * 30.0

    if method == 'Contiguous':
        if fr == 30:
            video_to_frames_contig(orig_id, patient_id, df_rows, cap, write_path=write_path)
        else:
            video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, write_path=write_path)
    else:
        raise Exception('Stride method not yet implemented!')


flow = cfg['PARAMS']['FLOW']

if not (flow == 'Yes'):
    input_folder = cfg['PATHS']['EXTRA_MASKED_VIDEOS']
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
        fr = round(fr / 30.0) * 30.0  # Cast to nearest multiple of 30
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
    out_df = pd.DataFrame(df_rows, columns=['id', 'patient_id'])
    out_df.to_csv(csv_out_path, index=False)
if df_rows_flow:
    csv_out_path_flow = os.path.join(csv_out_folder, 'extra_flow_mini_clips.csv')
    out_df_flow = pd.DataFrame(df_rows_flow, columns=['id', 'patient_id'])
    out_df_flow.to_csv(csv_out_path_flow, index=False)
