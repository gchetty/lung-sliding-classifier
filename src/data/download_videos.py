import pandas as pd
import mysql.connector
import os
import urllib.request
import yaml
from utils import refresh_folder

# Load dictionary of constants stored in config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']


def download(df, sliding, video_out_root_folder=cfg['PATHS']['UNMASKED_VIDEOS'], csv_out_folder=cfg['PATHS']['CSVS_OUTPUT']):
    '''
    Downloads ultrasound videos from the database in .mp4 format, and saves .csvs for tracing their metadata.
    :param df: A Pandas DataFrame which is the result of a specific query to the database. This is the saved .csv.
    :param sliding: A boolean for whether df is holding information on sliding or non_sliding clips.
    :param video_out_root_folder: The folder path for outputting the downloaded videos.
    :param csv_out_folder: The folder path for outputting the .csv.

    '''
    # Obtain the fraction of data we want from the database, according to config
    fraction = 0.0
    sliding_str = ''
    if sliding:
        sliding_str = 'sliding'
        fraction = cfg['PARAMS']['SLIDING_PROPORTION']
    else:
        sliding_str = 'no_sliding'
        fraction = cfg['PARAMS']['NO_SLIDING_PROPORTION']
    
    num_rows = int(fraction*len(df))
    df = df[:num_rows]

    # Make csv folder if it doesn't exist
    if not os.path.exists(csv_out_folder):
            os.makedirs(csv_out_folder)

    csv_out_path = os.path.join(csv_out_folder, sliding_str + '.csv')
    print('Writing to ' + csv_out_path + "...")
    df.to_csv(csv_out_path, index=False)

    # Make video root folder if it doesn't exist
    if not os.path.exists(video_out_root_folder):
            os.makedirs(video_out_root_folder)
    
    # Ensure all contents in the video folder are deleted, but that the folder exists
    video_out_folder = os.path.join(video_out_root_folder, sliding_str + '/')
    refresh_folder(video_out_folder)

    print('Writing ' + str(num_rows) + ' videos to ' + video_out_folder + '...')

    # Iterate through df to download the videos and preserve their id in the name
    for index, row in df.iterrows():
        urllib.request.urlretrieve(row['s3_path'], video_out_folder + row['id'] + '.mp4')

# Get database configs
USERNAME = cfg['DATABASE']['USERNAME']
PASSWORD = cfg['DATABASE']['PASSWORD']
HOST = cfg['DATABASE']['HOST']
DATABASE = cfg['DATABASE']['DATABASE']

# Establish connection to database
cnx = mysql.connector.connect(user=USERNAME, password=PASSWORD,
                              host=HOST,
                              database=DATABASE)

# Query database for sliding and no_sliding labeled data
sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings is 
                            null OR pleural_line_findings='thickened') and labelled = 1;''', 
                            cnx)

no_sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings='absent_lung_sliding' OR 
                               pleural_line_findings='thickened|absent_lung_sliding') and labelled = 1;''', 
                               cnx)

# If we're just asking the amount of videos available, the program will terminate after logging this information
JUST_ASKING_AMOUNT = cfg['PARAMS']['JUST_ASKING_AMOUNT']
if JUST_ASKING_AMOUNT:
    print('JUST_ASKING_AMOUNT is set to True in the config file; set this to False if you wanted to download the videos.' )
    print('In total, there are ' + str(len(sliding_df)) + ' available videos with sliding,' + 
      ' and ' + str(len(no_sliding_df)) + ' without.')
    exit()

print('In total, there are ' + str(len(sliding_df)) + ' available videos with sliding,' + 
      ' and ' + str(len(no_sliding_df)) + ' without.')

# Call download with each dataframe
download(sliding_df, sliding=True)
download(no_sliding_df, sliding=False)