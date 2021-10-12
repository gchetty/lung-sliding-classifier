import os
import sys
import pandas as pd
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


# helper for below
# from a given row index, finds nearest place where subsequent entries have different patient ids
def find_patient_division(df, index):
    counter = 0

    while True:

        if not (df.iloc[index - counter]['patient_id'] == df.iloc[index - counter + 1]['patient_id']):
            return index - counter + 1

        if not (df.iloc[index + counter]['patient_id'] == df.iloc[index + counter + 1]['patient_id']):
            return index + counter + 1

        counter += 1


# e.g. train = 0.8, val = 0.1, test = 0.1
# column is just added for labels, same df as input
def df_splits(df, train, val, test):
    if not (train + val + test == 1.0):
        return

    mark1 = int(len(df) * train)
    mark1 = find_patient_division(df, mark1 - 1)  # -1 because 0-indexed

    mark2 = int(len(df) * val + mark1)
    mark2 = find_patient_division(df, mark2 - 1)

    # add column & return df (0, 1, 2 --> train, val, test)
    split_labels = [0] * mark1 + [1] * (mark2 - mark1) + [2] * (len(df) - mark2)
    df['split'] = split_labels
    return True


# Import split params and check validity
train_prop = cfg['TRAIN']['SPLITS']['TRAIN']
val_prop = cfg['TRAIN']['SPLITS']['VAL']
test_prop = cfg['TRAIN']['SPLITS']['TEST']

if (train_prop + val_prop + test_prop) > 1.0:
    print('Invalid splits given in config file')
    sys.exit()

# CSV paths
csv_path = os.path.join(os.getcwd(), 'data/', cfg['PREPROCESS']['PATHS']['CSVS_OUTPUT'])
sliding_path = os.path.join(csv_path, 'sliding_mini_clips.csv')
no_sliding_path = os.path.join(csv_path, 'no_sliding_mini_clips.csv')

# Input csvs of mini-clips
sliding_df = pd.read_csv(sliding_path)
sliding_df = sliding_df.sort_values(by=['patient_id'])
no_sliding_df = pd.read_csv(no_sliding_path)
no_sliding_df = no_sliding_df.sort_values(by=['patient_id'])

# Determine splits, add to previously declared dataframes
df_splits(sliding_df, train_prop, val_prop, test_prop)
df_splits(no_sliding_df, train_prop, val_prop, test_prop)

# Write dataframes to csv
csv_dir = cfg['TRAIN']['PATHS']['CSVS']
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

sliding_out_path = os.path.join(csv_dir, 'sliding_splits.csv')
sliding_df.to_csv(sliding_out_path, index=False)

no_sliding_out_path = os.path.join(csv_dir, 'no_sliding_splits.csv')
no_sliding_df.to_csv(no_sliding_out_path, index=False)
