'''
Script for training experiments, including model training, hyperparameter search, cross validation, etc
'''

import yaml
import os
import pandas as pd
import tensorflow as tf

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))['TRAIN']


def train_model(model_def, preprocessing_fn, hparams, save_weights=False, log_dir=None, verbose=True):
    # Read in train, val, test dataframes
    CSVS_FOLDER = cfg['PATHS']['CSVS']
    train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'train.csv'))
    val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'val.csv'))
    test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'test.csv'))

    # Create TF datasets for training, validation and test sets
    train_set = tf.data.Dataset.from_tensor_slices((train_df['filename'].tolist(), train_df['label']))
    val_set = tf.data.Dataset.from_tensor_slices((val_df['filename'].tolist(), val_df['label']))
    test_set = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label']))

    print('no errors!')

train_model(None, None, None)
