import cv2
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from utils import refresh_folder
from tqdm import tqdm

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']

# PIPELINE UPDATE FOR BOX AND ORIG DIMS NOT APPLIED FOR FLOW!!!


def video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, seq_length=cfg['PARAMS']['WINDOW'],
                                resize=cfg['PARAMS']['IMG_SIZE'], write_path='', box=None,
                                base_fr=cfg['PARAMS']['BASE_FR']):

    '''
    Converts a LUS video file to mini-clips downsampled to a specified frame rate with specified sequence length

    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param cap: Captured video of full clip, returned by cv2.VideoCapture()
    :param fr: Frame rate (integer) of original clip - MUST be divisible by downsampled frame rate
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    :param box: Tuple of (ymin, xmin, ymax, xmax) of pleural line ROI
    :param base_fr: Base frame rate to downsample to
    '''

    # Check validity of frame rate param
    assert(isinstance(fr, int))
    assert(fr % base_fr == 0)

    frames = []
    stride = fr // base_fr

    index = 0  # Position in 'frames' array where the next frame is to be appended

    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if cap_width == 0 or cap_height == 0:
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index == 0:  # Add every nth frame only
                frame = cv2.resize(frame, tuple(reversed(resize)))
                weights = [0.2989, 0.5870, 0.1140]  # In accordance with tfa rgb to grayscale
                frame = np.dot(frame, weights).astype(np.uint8)
                frame = np.expand_dims(frame, axis=-1)
                frames.append(frame)
            index = (index + 1) % stride

    finally:
        cap.release()

    # Assemble and save mini-clips from extracted frames
    counter = 1
    num_mini_clips = len(frames) // seq_length
    for i in range(num_mini_clips):
        df_rows.append([orig_id + '_' + str(counter), patient_id])
        np.savez_compressed(write_path + '_' + str(counter), frames=frames[i * seq_length:i * seq_length + seq_length],
                            bounding_box=box, height_width=(cap_height, cap_width))
        counter += 1

    return


def video_to_frames_contig(orig_id, patient_id, df_rows, cap, seq_length=cfg['PARAMS']['WINDOW'],
                           resize=cfg['PARAMS']['IMG_SIZE'], write_path='', box=None):

    '''
    Converts a LUS video file to contiguous-frame mini-clips with specified sequence length

    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param cap: Captured video of full clip, returned by cv2.VideoCapture()
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    :param box: Tuple of (ymin, xmin, ymax, xmax) of pleural line ROI
    '''

    counter = seq_length
    mini_clip_num = 1  # nth mini-clip being made from the main clip
    frames = []

    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if cap_width == 0 or cap_height == 0:
        return

    try:
        while True:

            # When seq_length frames have been read, update df rows and write npz files
            # The id of the xth mini-clip from a main clip is the id of the main clip with _x appended to it
            if counter == 0:
                df_rows.append([orig_id + '_' + str(mini_clip_num), patient_id])  # append to what will make output dataframes
                np.savez_compressed(write_path + '_' + str(mini_clip_num), frames=frames,
                                    bounding_box=box, height_width=(cap_height, cap_width))  # output
                counter = seq_length
                mini_clip_num += 1
                frames = []

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, tuple(reversed(resize)))

            weights = [0.2989, 0.5870, 0.1140]  # In accordance with tfa rgb to grayscale
            frame = np.dot(frame, weights).astype(np.uint8)
            frame = np.expand_dims(frame, axis=-1)

            frames.append(frame)

            counter -= 1

    finally:
        cap.release()

    return


def flow_frames_to_npz_downsampled(path, orig_id, patient_id, df_rows, fr, seq_length=cfg['PARAMS']['WINDOW'],
                                   resize=cfg['PARAMS']['IMG_SIZE'], write_path='', base_fr=cfg['PARAMS']['BASE_FR']):

    '''
    Converts a directory of x and y flow frames to mini-clips, downsampled, with flows stacked on axis = -1

    :param path: Path to directory containing flow frames
    :param orig_id: ID of the LUS video file associated with the flow frames
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param fr: Frame rate (in FPS) of original LUS clip
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    :param base_fr: Base frame rate to downsample to
    '''

    frames_x = []
    frames_y = []

    stride = fr // base_fr

    # Read all flow frames
    for file in os.listdir(path):

        frame = cv2.imread(os.path.join(path, file), 0)
        frame = cv2.resize(frame, tuple(reversed(resize)))

        # NOT TESTED
        weights = [0.2989, 0.5870, 0.1140]  # In accordance with tfa rgb to grayscale
        frame = np.dot(frame, weights).astype(np.uint8)
        frame = np.expand_dims(frame, axis=-1)

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
        np.savez_compressed(write_path + '_' + str(counter), frames=np.stack((x_seq, y_seq), axis=-1))
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
        frame = cv2.resize(frame, tuple(reversed(resize)))

        # NOT TESTED
        weights = [0.2989, 0.5870, 0.1140]  # In accordance with tfa rgb to grayscale
        frame = np.dot(frame, weights).astype(np.uint8)
        frame = np.expand_dims(frame, axis=-1)

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
        np.savez_compressed(write_path + '_' + str(counter), frames=np.stack((x_seq, y_seq), axis=-1))
        counter += 1

    return


def video_to_npz(path, orig_id, patient_id, df_rows, write_path='', fr=None, box=None,
                 base_fr=cfg['PARAMS']['BASE_FR']):

    '''
    Converts a LUS video file to mini-clips
  
    :param path: Path to video file to be converted
    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param write_path: Path to directory where output mini-clips are saved
    :param fr: Frame rate of input
    :param box: Tuple of (ymin, xmin, ymax, xmax) of pleural line ROI
    :param base_fr: Base frame rate to downsample to
    '''
  
    cap = cv2.VideoCapture(path)

    if not fr:
        fr = round(cap.get(cv2.CAP_PROP_FPS))
        # Disregard clips with undesired frame rate, only if frame rate not passed in (passed in = override checking)
        if not (fr % base_fr == 0):
            return
    else:  # If frame rate is passed, cast frame rate to closest multiple of base frame rate
        fr = round(fr / base_fr) * base_fr

    if fr == base_fr:
        video_to_frames_contig(orig_id, patient_id, df_rows, cap, write_path=write_path, box=box)
    else:
        video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, write_path=write_path, box=box)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract mini-clips as npzs")
    parser.add_argument('--flow', default='False', type=str)
    parser.add_argument('--smooth', default='False', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    flow = True if (args.flow == 'True') else False
    smooth = True if (args.smooth == 'True') else False

    # Paths for video or frame input
    input_folder = ''
    if flow:
        input_folder = cfg['PATHS']['FLOW_VIDEOS']
    elif smooth:
        input_folder = cfg['PATHS']['SMOOTHED_VIDEOS']
    else:
        input_folder = cfg['PATHS']['MASKED_VIDEOS']
    sliding_input = os.path.join(input_folder, 'sliding/')
    no_sliding_input = os.path.join(input_folder, 'no_sliding/')

    # Paths for npz output
    npz_folder = ''

    if not flow:
        npz_folder = cfg['PATHS']['NPZ']
    else:
        npz_folder = cfg['PATHS']['FLOW_NPZ']

    refresh_folder(npz_folder)

    sliding_npz_folder = os.path.join(npz_folder, 'sliding/')
    os.makedirs(sliding_npz_folder)

    no_sliding_npz_folder = os.path.join(npz_folder, 'no_sliding/')
    os.makedirs(no_sliding_npz_folder)

    # Each element is (mini-clip_id, patient_id) for download as csv
    df_rows_sliding = []
    df_rows_no_sliding = []

    # Load original csvs for patient ids and frame rate csvs
    csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']

    sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'sliding.csv'))
    no_sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding.csv'))

    sliding_fps_df = pd.read_csv(os.path.join(csv_out_folder, 'sliding_frame_rates.csv'))
    no_sliding_fps_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding_frame_rates.csv'))

    # Load object detection csv for m-mode
    use_box = cfg['PARAMS']['USE_BOUNDING_BOX']
    if use_box:
        sliding_box_df = pd.read_csv(os.path.join(csv_out_folder, 'sliding_boxes.csv'))
        no_sliding_box_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding_boxes.csv'))

    base_fr = cfg['PARAMS']['BASE_FR']

    # Iterate through clips and extract & download mini-clips

    if flow:

        for id in os.listdir(sliding_input):
            path = os.path.join(sliding_input, id)
            patient_id = ((sliding_df[sliding_df['id'] == id])['patient_id']).values[0]
            fr = ((sliding_fps_df[sliding_fps_df['id'] == id])['frame_rate']).values[0]
            if fr == base_fr:
                flow_frames_to_npz_contig(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_sliding,
                                          write_path=(sliding_npz_folder + id))
            else:
                flow_frames_to_npz_downsampled(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_sliding, fr=fr,
                                               write_path=(sliding_npz_folder + id))

        for id in os.listdir(no_sliding_input):
            path = os.path.join(no_sliding_input, id)
            patient_id = ((no_sliding_df[no_sliding_df['id'] == id])['patient_id']).values[0]
            fr = ((no_sliding_fps_df[no_sliding_fps_df['id'] == id])['frame_rate']).values[0]
            if fr == base_fr:
                flow_frames_to_npz_contig(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_no_sliding,
                                          write_path=(no_sliding_npz_folder + id))
            else:
                flow_frames_to_npz_downsampled(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_no_sliding,
                                               fr=fr, write_path=(no_sliding_npz_folder + id))

    else:

        for file in tqdm(os.listdir(sliding_input)):

            f = os.path.join(sliding_input, file)
            patient_id = ((sliding_df[sliding_df['id'] == file[:-4]])['patient_id']).values[0]

            if use_box:
                query = sliding_box_df[sliding_box_df['id'] == file[:-4]]

                if not query.empty:
                    box_info = query.iloc[0]
                    box = (box_info['ymin'], box_info['xmin'], box_info['ymax'], box_info['xmax'])

                    video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_sliding,
                                 write_path=(sliding_npz_folder + file[:-4]), box=box)
            else:
                video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_sliding,
                             write_path=(sliding_npz_folder + file[:-4]))

        for file in tqdm(os.listdir(no_sliding_input)):

            f = os.path.join(no_sliding_input, file)
            patient_id = ((no_sliding_df[no_sliding_df['id'] == file[:-4]])['patient_id']).values[0]

            if use_box:
                query = no_sliding_box_df[no_sliding_box_df['id'] == file[:-4]]

                if not query.empty:
                    box_info = query.iloc[0]
                    box = (box_info['ymin'], box_info['xmin'], box_info['ymax'], box_info['xmax'])

                    video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_no_sliding,
                                 write_path=(no_sliding_npz_folder + file[:-4]), box=box)
            else:
                video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_no_sliding,
                             write_path=(no_sliding_npz_folder + file[:-4]))

    # Download dataframes linking mini-clip ids and patient ids as csv files
    out_df_sliding = pd.DataFrame(df_rows_sliding, columns=['id', 'patient_id'])
    out_df_no_sliding = pd.DataFrame(df_rows_no_sliding, columns=['id', 'patient_id'])

    csv_out_path_sliding = ''
    csv_out_path_no_sliding = ''

    if flow:
        csv_out_path_sliding = os.path.join(csv_out_folder, 'sliding_flow_mini_clips.csv')
        csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_flow_mini_clips.csv')
    else:
        csv_out_path_sliding = os.path.join(csv_out_folder, 'sliding_mini_clips.csv')
        csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_mini_clips.csv')

    out_df_sliding.to_csv(csv_out_path_sliding, index=False)
    out_df_no_sliding.to_csv(csv_out_path_no_sliding, index=False)
