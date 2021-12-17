import pandas as pd
import mysql.connector
import os
import urllib.request
import yaml
import cv2

# Load dictionary of constants stored in config.yml & db credentials in database_config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']


def download(df, sliding, fr_rows, video_out_root_folder=cfg['PATHS']['UNMASKED_VIDEOS'],
             csv_out_folder=cfg['PATHS']['CSVS_OUTPUT']):
    '''
    Downloads ultrasound videos from the database in .mp4 format, and saves .csvs for tracing their metadata.

    :param df: A Pandas DataFrame which is the result of a specific query to the database. This is the saved .csv.
    :param sliding: A boolean for whether df is holding information on sliding or non_sliding clips.
    :param fr_rows: Dataframe of (clip id, frame rate)
    :param video_out_root_folder: The folder path for outputting the downloaded videos.
    :param csv_out_folder: The folder path for outputting the .csv.
    '''

    # Optionally sort, then shuffle the df (and therefore the saved csv) according to config
    shuffle = cfg['PARAMS']['SHUFFLE']

    if shuffle:
        df = df.sort_values(['id'])
        df = df.sample(frac=1, random_state=cfg['PARAMS']['RANDOM_SEED'])

    # Obtain the fraction of data we want from the database, according to config
    if sliding:
        sliding_str = 'sliding'
    else:
        sliding_str = 'no_sliding'

    # Exit if csv folder doesn't exist
    if not os.path.exists(csv_out_folder):
        return

    # Add rows to existing csv
    csv_out_path = os.path.join(csv_out_folder, sliding_str + '.csv')
    print('Writing to ' + csv_out_path + "...")
    df_orig = pd.read_csv(csv_out_path)
    df_orig = pd.concat([df_orig, df])
    df_orig.to_csv(csv_out_path, index=False)

    # Make video root folder if it doesn't exist
    if not os.path.exists(video_out_root_folder):
        os.makedirs(video_out_root_folder)

    # Ensure all video output folder exists (DO NOT empty it)
    video_out_folder = os.path.join(video_out_root_folder, sliding_str + '/')
    if not os.path.exists(video_out_folder):
        os.makedirs(video_out_folder)

    print('Writing ' + str(len(df)) + ' videos to ' + video_out_folder + '...')

    # Iterate through df to download the videos and preserve their id in the name
    for index, row in df.iterrows():
        out_path = video_out_folder + row['id'] + '.mp4'
        urllib.request.urlretrieve(row['s3_path'], out_path)

        # Get frame rate
        cap = cv2.VideoCapture(out_path)
        fr = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # Discard video if frame rate is not multiple of 30
        if not (fr % 30 == 0):
            os.remove(out_path)
        else:
            # Add to frame rate CSV rows
            fr_rows.append([row['id'], fr])


df = pd.read_csv('no_lung_sliding_new_sprints_2021-12-09.csv')

# Store rows for frame rate csv
fr_no_sliding_rows = []

# Call download with dataframe
download(df, sliding=False, fr_rows=fr_no_sliding_rows)

# Append to existing frame rate CSV
csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']

out_df_no_sliding = pd.DataFrame(fr_no_sliding_rows, columns=['id', 'frame_rate'])
csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_frame_rates.csv')
orig_df = pd.read_csv(csv_out_path_no_sliding)
out_df_no_sliding = pd.concat([orig_df, out_df_no_sliding])
out_df_no_sliding.to_csv(csv_out_path_no_sliding, index=False)
