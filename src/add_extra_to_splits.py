import os
import sys
import pandas as pd
import yaml
import random
from sklearn.model_selection import train_test_split

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def df_splits_two_stream(df, flow_df, train, val, test, random_state=cfg['TRAIN']['SPLITS']['RANDOM_SEED']):
    '''
    Splits dataframe into train, val, and test splits, selecting only clips present in both input dataframes

    :param df: Dataframe of regular mini-clip ids
    :param flow_df: Dataframe of optical flow mini-clip ids
    :param train: 0 < Float < 1 representing the training fraction
    :param val:   0 < Float < 1 representing the validation fraction
    :param test:  0 < Float < 1 representing the test fraction
    :param random_state: An integer for setting the random seed

    :return: Dataframe with split labels
    '''

    # Keep only mini-clips existing for regular frames and flow frames in df
    drop_indices = []
    for index, row in df.iterrows():
        if not (row['id'] in list(flow_df['id'])):
            drop_indices.append(index)
    df = df.drop(drop_indices)

    # Make splits from remaining mini-clips
    df = df_splits(df, train, val, test, random_state)

    return df


def df_splits(df, train, val, test, random_state=cfg['TRAIN']['SPLITS']['RANDOM_SEED']):
    '''
    Splits the dataframe into train, val and test (adds a split column with train=0, val=1, test=2).
    Does not use patient id (intended for unprocessed/undocumented clips)

    :param df: The sliding or no_sliding dataframe to adjust
    :param train: 0 < Float < 1 representing the training fraction
    :param val:   0 < Float < 1 representing the validation fraction
    :param test:  0 < Float < 1 representing the test fraction
    :param random_state: An integer for setting the random seed

    :return: Dataframe with split labels
    '''

    if random_state == -1:
        random_state = random.randint(0, 1000)

    ids = df['id'].to_numpy()

    if test == 0:
        splits = train_test_split(ids, train_size=train, random_state=random_state)
        splits.append([])
    else:
        splits = train_test_split(ids, train_size=train, random_state=random_state)
        val_test_splits = train_test_split(splits[1], train_size=(val / (val + test)), shuffle=False)
        splits[1] = val_test_splits[0]
        splits.append(val_test_splits[1])

    # Add split labels to dataframe
    split_labels = []
    for index, row in df.iterrows():
        curr_id = row['id']
        if curr_id in splits[1]:
            split_labels.append(1)
        elif curr_id in splits[2]:
            split_labels.append(2)
        else:
            split_labels.append(0)

    df['split'] = split_labels

    return df


# Import split params and check validity
train_prop = cfg['TRAIN']['SPLITS']['TRAIN']
val_prop = cfg['TRAIN']['SPLITS']['VAL']
test_prop = cfg['TRAIN']['SPLITS']['TEST']

s = sum([train_prop, val_prop, test_prop])
if (s > 1.0) or (train_prop < 0) or (val_prop < 0) or (test_prop < 0):
    print('Invalid splits given in config file')
    sys.exit()

# CSV paths
csv_path = os.path.join(os.getcwd(), 'data/', cfg['PREPROCESS']['PATHS']['CSVS_OUTPUT'])

flow = cfg['PREPROCESS']['PARAMS']['FLOW']

regular_path = ''
flow_path = ''

if not (flow == 'No'):
    flow_path = os.path.join(csv_path, 'extra_flow_mini_clips.csv')
if not (flow == 'Yes'):
    regular_path = os.path.join(csv_path, 'extra_mini_clips.csv')

# Read mini-clip path
if regular_path:
    regular_df = pd.read_csv(regular_path)
if flow_path:
    flow_df = pd.read_csv(flow_path)

# Determine splits, add to previously declared dataframes
if flow == 'No':
    regular_df = df_splits(regular_df, train_prop, val_prop, test_prop)
elif flow == 'Yes':
    regular_df = df_splits(flow_df, train_prop, val_prop, test_prop)
else:
    regular_df = df_splits_two_stream(regular_df, flow_df, train_prop, val_prop, test_prop)

# Add label to dataframes
l0 = [0] * len(regular_df)
regular_df['label'] = l0

# Add file path to dataframes
npz_dir = ''
if flow == 'Yes':
    npz_dir = cfg['PREPROCESS']['PATHS']['EXTRA_FLOW_NPZ']
else:
    npz_dir = cfg['PREPROCESS']['PATHS']['EXTRA_NPZ']

paths = []
for index, row in regular_df.iterrows():
    paths.append(os.path.join(os.getcwd(), 'data/', npz_dir, row['id'] + '.npz'))
regular_df['filename'] = paths

if flow == 'Both':

    npz_dir = cfg['PREPROCESS']['PATHS']['EXTRA_FLOW_NPZ']

    paths = []
    for index, row in regular_df.iterrows():
        paths.append(os.path.join(os.getcwd(), 'data/', npz_dir, row['id'] + '.npz'))
    regular_df['flow_filename'] = paths

# Print the proportion of added mini-clips to each split
print('Train: {}'.format(len(regular_df[regular_df['split']==0])))
print('Val: {}'.format(len(regular_df[regular_df['split']==1])))
print('Test: {}'.format(len(regular_df[regular_df['split']==2])))

# Shuffle
final_df = regular_df.sample(frac=1, random_state=2)

# Split into train_df, val_df, test_df
train_df = final_df[final_df['split']==0]
val_df = final_df[final_df['split']==1]
test_df = final_df[final_df['split']==2]

# Read existing split csvs and add to them
csv_dir = cfg['TRAIN']['PATHS']['CSVS']

train_df_path = ''
val_df_path = ''
test_df_path = ''

if flow == 'Yes':
    train_df_path = os.path.join(csv_dir, 'flow_train.csv')
    val_df_path = os.path.join(csv_dir, 'flow_val.csv')
    test_df_path = os.path.join(csv_dir, 'flow_test.csv')
else:
    train_df_path = os.path.join(csv_dir, 'train.csv')
    val_df_path = os.path.join(csv_dir, 'val.csv')
    test_df_path = os.path.join(csv_dir, 'test.csv')

existing_train_df = pd.read_csv(train_df_path)
existing_val_df = pd.read_csv(val_df_path)
existing_test_df = pd.read_csv(test_df_path)

random_state = cfg['TRAIN']['SPLITS']['RANDOM_SEED']

if random_state == -1:
    random_state = random.randint(0, 1000)

train_df = pd.concat([existing_train_df, train_df]).sample(frac=1, random_state=random_state)
val_df = pd.concat([existing_val_df, val_df]).sample(frac=1, random_state=random_state)
test_df = pd.concat([existing_test_df, test_df]).sample(frac=1, random_state=random_state)

# Write to csvs
train_df.to_csv(train_df_path, index=False)
val_df.to_csv(val_df_path, index=False)
test_df.to_csv(test_df_path, index=False)
