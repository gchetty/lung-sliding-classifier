import pandas as pd
import mysql.connector
import os
import urllib.request
import yaml
import cv2
from utils import refresh_folder

# Load dictionary of constants stored in config.yml & db credentials in database_config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']
database_cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../database_config.yml"), 'r'))


def download(df, sliding, fr_rows, video_out_root_folder=cfg['PATHS']['UNMASKED_VIDEOS'],
             csv_out_folder=cfg['PATHS']['CSVS_OUTPUT'], base_fr=cfg['PARAMS']['BASE_FR']):

    '''
    Downloads ultrasound videos from the database in .mp4 format, and saves .csvs for tracing their metadata.

    :param df: A Pandas DataFrame which is the result of a specific query to the database. This is the saved .csv.
    :param sliding: A boolean for whether df is holding information on sliding or non_sliding clips.
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
        out_path = video_out_folder + row['id'] + '.mp4'
        urllib.request.urlretrieve(row['s3_path'], out_path)

        # Get frame rate
        cap = cv2.VideoCapture(out_path)
        fr = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # Discard video if frame rate is not multiple of base frame rate
        if not (fr % base_fr == 0):
            os.remove(out_path)
        else:
            # Add to frame rate CSV rows
            fr_rows.append([row['id'], fr])


# Get database configs
USERNAME = database_cfg['USERNAME']
PASSWORD = database_cfg['PASSWORD']
HOST = database_cfg['HOST']
DATABASE = database_cfg['DATABASE']

# Establish connection to database
cnx = mysql.connector.connect(user=USERNAME, password=PASSWORD,
                              host=HOST,
                              database=DATABASE)

# Query database for sliding and no_sliding labeled data
# sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings is
#                             null OR pleural_line_findings='thickened') and labelled = 1 and view = 'parenchymal';''',
#                             cnx)
# no_sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings='absent_lung_sliding' OR
#                                pleural_line_findings='thickened|absent_lung_sliding') and labelled = 1 and
#                                view = 'parenchymal';''',
#                                cnx)

# # Query Database for sliding and no_sliding labeled data but excluding significant probe movement tag
# sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings is null OR
#                             pleural_line_findings='thickened') AND (quality NOT LIKE '%significant_probe_movement%' OR
#                             quality is null) and labelled = 1 and view = 'parenchymal';''', cnx)
# no_sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings='absent_lung_sliding' OR
#                                pleural_line_findings='thickened|absent_lung_sliding') AND
#                                (quality NOT LIKE '%significant_probe_movement%' OR quality is null) and
#                                labelled = 1 and view = 'parenchymal';''', cnx)

# Query Database for sliding and no_sliding labeled data but excluding significant probe movement and only A lines for
# sliding class
# sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings is null OR
#                             pleural_line_findings='thickened') AND (quality NOT LIKE '%significant_probe_movement%' OR
#                             quality is null) AND (a_or_b_lines='a_lines') and labelled = 1 and
#                             view = 'parenchymal';''', cnx)
# no_sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings='absent_lung_sliding' OR
#                                pleural_line_findings='thickened|absent_lung_sliding') AND
#                                (quality NOT LIKE '%significant_probe_movement%' OR quality is null) and
#                                labelled = 1 and view = 'parenchymal';''', cnx)

# Query Database for sliding and no_sliding labeled data but excluding significant probe movement and B lines
# and pleural effusion or consolidation
sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings is null OR
                            pleural_line_findings='thickened') AND (quality NOT LIKE '%significant_probe_movement%' OR
                            quality is null) AND (a_or_b_lines='a_lines') AND (pleural_effusion is null) 
                            AND (consolidation is null) and labelled = 1 and view = 'parenchymal';''', cnx)
# cnx = mysql.connector.connect(user=USERNAME, password=PASSWORD,
#                               host=database_cfg['EXTERNAL_HOST'],
#                               database=DATABASE)
# chile_sliding_df = pd.read_sql('''SELECT * FROM clips_chile WHERE view = 'parenchymal';''', cnx)
no_sliding_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings='absent_lung_sliding' OR
                               pleural_line_findings='thickened|absent_lung_sliding') AND
                               (quality NOT LIKE '%significant_probe_movement%' OR quality is null) AND 
                               (a_or_b_lines='a_lines') AND (pleural_effusion is null) AND (consolidation is null)
                               and labelled = 1 and view = 'parenchymal';''', cnx)

# Query database for extra negative examples from sprints
no_sliding_extra_df = pd.read_sql('''SELECT * FROM clips WHERE (pleural_line_findings='absent_lung_sliding'
                                      OR pleural_line_findings='thickened|absent_lung_sliding') AND
                               (quality NOT LIKE '%significant_probe_movement%' OR quality is null) AND 
                               (a_or_b_lines='a_lines') AND (pleural_effusion is null) AND (consolidation is null) 
                               AND labelbox_project_number LIKE 'Lung sliding sprint%';''', cnx)

# Load examples that must be excluded from negative class (bad clips)
bad_no_sliding_df = pd.read_csv(os.path.join(cfg['PATHS']['CSVS_OUTPUT'], 'bad_ids.csv'))

no_sliding_df = pd.concat([no_sliding_df, no_sliding_extra_df])
no_sliding_df = no_sliding_df[~no_sliding_df['id'].isin(bad_no_sliding_df['id'])]

# If we're just asking the amount of videos available, the program will terminate after logging this information
AMOUNT_ONLY = cfg['PARAMS']['AMOUNT_ONLY']
if AMOUNT_ONLY:
    print('AMOUNT_ONLY is set to True in the config file; set this to False if you wanted to download the videos.' )
    print('In total, there are ' + str(len(sliding_df)) + ' available videos with sliding,' +
      ' and ' + str(len(no_sliding_df)) + ' without.')
    exit()

print('In total, there are ' + str(len(sliding_df)) + ' available videos with sliding,' +
      ' and ' + str(len(no_sliding_df)) + ' without.')

# Store rows for frame rate csv
fr_sliding_rows = []
fr_no_sliding_rows = []

# Call download with each dataframe
download(sliding_df, sliding=True, fr_rows=fr_sliding_rows)
download(no_sliding_df, sliding=False, fr_rows=fr_no_sliding_rows)

# Save frame rate CSVs
csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']

out_df_sliding = pd.DataFrame(fr_sliding_rows, columns=['id', 'frame_rate'])
csv_out_path_sliding = os.path.join(csv_out_folder, 'sliding_frame_rates.csv')
out_df_sliding.to_csv(csv_out_path_sliding, index=False)

out_df_no_sliding = pd.DataFrame(fr_no_sliding_rows, columns=['id', 'frame_rate'])
csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_frame_rates.csv')
out_df_no_sliding.to_csv(csv_out_path_no_sliding, index=False)
