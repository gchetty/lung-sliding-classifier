# Temporary script to only convert absent sliding examples to npz

import cv2
import numpy as np
import pandas as pd
import os
import yaml
import argparse

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']


def video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, seq_length=cfg['PARAMS']['WINDOW'],
                                resize=cfg['PARAMS']['IMG_SIZE'], write_path='', crop=True):
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
    :param crop: Boolean, Whether inputs are cropped to pleural line or not
    '''

    # Check validity of frame rate param
    assert (isinstance(fr, int))
    assert (fr % 30 == 0)

    frames = []
    stride = fr // 30

    index = 0  # Position in 'frames' array where the next frame is to be appended

    if crop:

        cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if cap_width == 0 or cap_height == 0:
            return

        new_width = None
        new_height = None

        if cap_width > cap_height:
            new_width = tuple(resize)[0]
            height_resize = int((cap_height / cap_width) * new_width)
            pad_top = int((tuple(resize)[1] - height_resize) / 2)
            pad_bottom = tuple(resize)[1] - height_resize - pad_top
            pad_left, pad_right = 0, 0
        else:
            new_height = tuple(resize)[1]
            width_resize = int((cap_width / cap_height) * new_height)
            pad_left = int((tuple(resize)[0] - width_resize) / 2)
            pad_right = tuple(resize)[0] - width_resize - pad_left
            pad_top, pad_bottom = 0, 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index == 0:  # Add every nth frame only
                if crop:
                    if new_width:
                        frame = cv2.resize(frame, (new_width, height_resize))
                    else:
                        frame = cv2.resize(frame, (width_resize, new_height))
                    frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])
                else:
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
                           resize=cfg['PARAMS']['IMG_SIZE'], write_path='', crop=True):
    '''
    Converts a LUS video file to contiguous-frame mini-clips with specified sequence length

    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param cap: Captured video of full clip, returned by cv2.VideoCapture()
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    :param crop: Boolean, Whether inputs are cropped to pleural line or not
    '''

    counter = seq_length
    mini_clip_num = 1  # nth mini-clip being made from the main clip
    frames = []

    if crop:

        cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if cap_width == 0 or cap_height == 0:
            return

        new_width = None
        new_height = None

        if cap_width > cap_height:
            new_width = tuple(resize)[0]
            height_resize = int((cap_height / cap_width) * new_width)
            pad_top = int((tuple(resize)[1] - height_resize) / 2)
            pad_bottom = tuple(resize)[1] - height_resize - pad_top
            pad_left, pad_right = 0, 0
        else:
            new_height = tuple(resize)[1]
            width_resize = int((cap_width / cap_height) * new_height)
            pad_left = int((tuple(resize)[0] - width_resize) / 2)
            pad_right = tuple(resize)[0] - width_resize - pad_left
            pad_top, pad_bottom = 0, 0

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

            if crop:
                if new_width:
                    frame = cv2.resize(frame, (new_width, height_resize))
                else:
                    frame = cv2.resize(frame, (width_resize, new_height))
                frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])
            else:
                frame = cv2.resize(frame, tuple(resize))

            frames.append(frame)

            counter -= 1

    finally:
        cap.release()

    return


def flow_frames_to_npz_downsampled(path, orig_id, patient_id, df_rows, fr, seq_length=cfg['PARAMS']['WINDOW'],
                                   resize=cfg['PARAMS']['IMG_SIZE'], write_path='', crop=True):
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
    :param crop: Boolean, Whether inputs are cropped to pleural line or not
    '''

    frames_x = []
    frames_y = []

    stride = fr // 30

    # Read all flow frames
    for file in os.listdir(path):

        frame = cv2.imread(os.path.join(path, file), 0)

        if crop:

            orig_width = frame.shape[1]  # height dimension should be first
            orig_height = frame.shape[0]

            if orig_width > orig_height:
                new_width = tuple(resize)[0]
                height_resize = int((orig_height / orig_width) * new_width)
                pad_top = int((tuple(resize)[1] - height_resize) / 2)
                pad_bottom = tuple(resize)[1] - height_resize - pad_top
                pad_left, pad_right = 0, 0
                frame = cv2.resize(frame, (new_width, height_resize))
            else:
                new_height = tuple(resize)[1]
                width_resize = int((orig_width / orig_height) * new_height)
                pad_left = int((tuple(resize)[0] - width_resize) / 2)
                pad_right = tuple(resize)[0] - width_resize - pad_left
                pad_top, pad_bottom = 0, 0
                frame = cv2.resize(frame, (width_resize, new_height))

            frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                       value=[0, 0, 0])

        else:

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
                              resize=cfg['PARAMS']['IMG_SIZE'], write_path='', crop=True):
    '''
    Converts a directory of x and y flow frames to contiguous mini-clips, with flows stacked on axis = -1

    :param path: Path to directory containing flow frames
    :param orig_id: ID of the LUS video file associated with the flow frames
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    :param crop: Boolean, Whether inputs are cropped to pleural line or not
    '''

    frames_x = []
    frames_y = []

    # Read all flow frames
    for file in os.listdir(path):

        frame = cv2.imread(os.path.join(path, file), 0)

        if crop:

            orig_width = frame.shape[1]  # height dimension should be first
            orig_height = frame.shape[0]

            if orig_width > orig_height:
                new_width = tuple(resize)[0]
                height_resize = int((orig_height / orig_width) * new_width)
                pad_top = int((tuple(resize)[1] - height_resize) / 2)
                pad_bottom = tuple(resize)[1] - height_resize - pad_top
                pad_left, pad_right = 0, 0
                frame = cv2.resize(frame, (new_width, height_resize))
            else:
                new_height = tuple(resize)[1]
                width_resize = int((orig_width / orig_height) * new_height)
                pad_left = int((tuple(resize)[0] - width_resize) / 2)
                pad_right = tuple(resize)[0] - width_resize - pad_left
                pad_top, pad_bottom = 0, 0
                frame = cv2.resize(frame, (width_resize, new_height))

            frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                       value=[0, 0, 0])

        else:

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


def video_to_npz(path, orig_id, patient_id, df_rows, write_path='', method=cfg['PARAMS']['METHOD'], fr=None, crop=True):
    '''
    Converts a LUS video file to mini-clips

    :param path: Path to video file to be converted
    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param write_path: Path to directory where output mini-clips are saved
    :param method: Method of frame extraction for mini-clips, either 'Contiguous' or ' Stride'
    :param fr: Frame rate of input
    :param crop: Boolean, Whether inputs are cropped to pleural line or not
    '''

    cap = cv2.VideoCapture(path)

    if not fr:
        fr = round(cap.get(cv2.CAP_PROP_FPS))
        # Disregard clips with undesired frame rate, only if frame rate not passed in (passed in = override checking)
        if not (fr % 30 == 0):
            return
    else:  # If frame rate is passed, cast frame rate to closest multiple of 30
        fr = round(fr / 30.0) * 30.0

    if method == 'Contiguous':
        if fr == 30:
            video_to_frames_contig(orig_id, patient_id, df_rows, cap, write_path=write_path, crop=crop)
        else:
            video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, write_path=write_path, crop=crop)
    else:
        raise Exception('Stride method not yet implemented!')


def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--flow', default='False', type=str)
    parser.add_argument('--crop', default='False', type=str)
    args = parser.parse_args()
    return args


if __name__ == 'main':

    args = parse_args()

    flow = True if (args.flow == 'True') else False
    crop = True if (args.crop == 'True') else False

    # Paths for video or frame input
    input_folder = ''
    if flow:
        input_folder = cfg['PATHS']['FLOW_VIDEOS']
    elif crop:
        input_folder = cfg['PATHS']['CROPPED_VIDEOS']
    else:
        input_folder = cfg['PATHS']['MASKED_VIDEOS']
    no_sliding_input = os.path.join(input_folder, 'no_sliding/')

    # Paths for npz output
    npz_folder = ''

    if not flow:
        npz_folder = cfg['PATHS']['NPZ']
    else:
        npz_folder = cfg['PATHS']['FLOW_NPZ']

    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)

    no_sliding_npz_folder = os.path.join(npz_folder, 'no_sliding/')
    if not os.path.exists(no_sliding_npz_folder):
        os.makedirs(no_sliding_npz_folder)

    # Each element is (mini-clip_id, patient_id) for download as csv
    df_rows_no_sliding = []

    # Load original csvs for patient ids and frame rate csvs
    csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']

    no_sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding.csv'))

    no_sliding_fps_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding_frame_rates.csv'))

    # Iterate through clips and extract & download mini-clips

    if flow:

        for id in os.listdir(no_sliding_input):
            path = os.path.join(no_sliding_input, id)
            patient_id = ((no_sliding_df[no_sliding_df['id'] == id])['patient_id']).values[0]
            fr = ((no_sliding_fps_df[no_sliding_fps_df['id'] == id])['frame_rate']).values[0]
            if fr == 30:
                flow_frames_to_npz_contig(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_no_sliding,
                                          write_path=(no_sliding_npz_folder + id), crop=crop)
            else:
                flow_frames_to_npz_downsampled(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_no_sliding,
                                               fr=fr, write_path=(no_sliding_npz_folder + id), crop=crop)

    else:

        for file in os.listdir(no_sliding_input):
            f = os.path.join(no_sliding_input, file)
            patient_id = ((no_sliding_df[no_sliding_df['id'] == file[:-4]])['patient_id']).values[0]
            video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_no_sliding,
                         write_path=(no_sliding_npz_folder + file[:-4]), crop=crop)

    # Download dataframes linking mini-clip ids and patient ids as csv files
    out_df_no_sliding = pd.DataFrame(df_rows_no_sliding, columns=['id', 'patient_id'])

    csv_out_path_no_sliding = ''

    if flow:
        csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_flow_mini_clips.csv')
    else:
        csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_mini_clips.csv')

    # Append to existing rows (for older clips that we already have)
    orig_csv = pd.read_csv(csv_out_path_no_sliding)
    out_df_no_sliding = pd.concat([orig_csv, out_df_no_sliding])

    out_df_no_sliding.to_csv(csv_out_path_no_sliding, index=False)
