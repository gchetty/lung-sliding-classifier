import pandas as pd
import mysql.connector
import os
import urllib.request
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"config.yml"), 'r'))

USERNAME = cfg['DATABASE']['USERNAME']
PASSWORD = cfg['DATABASE']['PASSWORD']
HOST = cfg['DATABASE']['HOST']
DATABASE = cfg['DATABASE']['DATABASE']

cnx = mysql.connector.connect(user=USERNAME, password=PASSWORD,
                              host=HOST,
                              database=DATABASE)

sliding_df = pd.read_sql('''SELECT * FROM clips WHERE pleural_line_findings is 
                            null OR pleural_line_findings='thickened';''', 
                            cnx)

no_sliding_df = pd.read_sql('''SELECT * FROM clips WHERE pleural_line_findings='absent_lung_sliding' OR 
                               pleural_line_findings='thickened|absent_lung_sliding';''', 
                               cnx)

print('In total, there are ' + str(len(sliding_df)) + ' available videos with sliding,' + 
      ' and ' + str(len(no_sliding_df)) + ' without.')

def download(df, output_path, sliding=True):
    shuffle = cfg['PARAMS']['SHUFFLE']

    if shuffle:
        df = df.sample(frac=1, random_state=cfg['PARAMS']['RANDOM_SEED'])
    
    fraction = None
    if sliding:
        fraction = cfg['PARAMS']['SLIDING_PROPORTION']
    else:
        fraction = cfg['PARAMS']['NO_SLIDING_PROPORTION']
    
    num_rows = int(fraction*len(df))
    df = df[:num_rows]
    print('Writing ' + str(num_rows) + ' videos to ' + output_path + '...')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for index, row in df.iterrows():
        urllib.request.urlretrieve(row['s3_path'], output_path + row['id'] + '.mp4')

OUTPUT_PATH = cfg['PATHS']['OUTPUT']
sliding_path = os.path.join(OUTPUT_PATH, 'sliding/')
no_sliding_path = os.path.join(OUTPUT_PATH, 'no_sliding/')

download(sliding_df, sliding_path)
download(no_sliding_df, no_sliding_path)