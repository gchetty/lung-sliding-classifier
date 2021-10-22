'''
Script for training experiments, including model training, hyperparameter search, cross validation, etc
'''

import yaml
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow_addons.metrics import F1Score
from tensorflow_model_analysis.metrics import Specificity
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocessor import Preprocessor
from models.models import *

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def train_model(model_def_str=cfg['TRAIN']['MODEL_DEF'], 
                hparams=cfg['TRAIN']['PARAMS']['TEST1'],
                model_out_dir=cfg['TRAIN']['PATHS']['MODEL_OUT']):
    '''
    Trains and saves a model given specific hyperparameters
    :param model_def_str: A string in {'test1', ... } specifying the name of the desired model
    :param hparams: A dictionary specifying the hyperparameters to use
    :param model_out_dir: The path to save the model
    '''
    # Get the model function and preprocessing function 
    model_def_fn, preprocessing_fn = get_model(model_def_str)

    # Read in training, validation, and test dataframes
    CSVS_FOLDER = cfg['TRAIN']['PATHS']['CSVS']
    train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'train.csv'))
    val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'val.csv'))
    test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'test.csv'))

    # Create TF datasets for training, validation and test sets
    # Note: This does NOT load the dataset into memory! We specify paths,
    #       and labels so that TensorFlow can perform dynamic loading.
    train_set = tf.data.Dataset.from_tensor_slices((train_df['filename'].tolist(), train_df['label']))
    val_set = tf.data.Dataset.from_tensor_slices((val_df['filename'].tolist(), val_df['label']))
    test_set = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label']))

    # Create preprocessing object given the preprocessing function for model_def
    preprocessor = Preprocessor(preprocessing_fn)

    # Define the preprocessing pipelines for train, test and validation
    train_set = preprocessor.prepare(train_set, train_df, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, val_df, shuffle=False, augment=False)
    test_set = preprocessor.prepare(test_set, test_df, shuffle=False, augment=False)

    # Create dictionary for class weights:
    # Taken from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    num_no_sliding = len(train_df[train_df['label']==0])
    num_sliding = len(train_df[train_df['label']==1])
    total = num_no_sliding + num_sliding
    weight_for_0 = (1 / num_no_sliding) * (total / 2.0)
    weight_for_1 = (1 / num_sliding) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(class_weight)

    # Defining metrics (accuracy, AUC, F1, Precision, Recall)
    classes = [0, 1]
    n_classes = len(classes)
    #threshold = 1.0 / n_classes
    metrics = ['accuracy', AUC(name='auc'), F1Score(num_classes=1)]
    metrics += [Precision()]
    metrics += [Recall()]
    #metrics += [Specificity()]

    # Creating a ModelCheckpoint for saving the model
    save_cp = ModelCheckpoint(model_out_dir, save_best_only=cfg['TRAIN']['SAVE_BEST_ONLY'])

    # Get the model
    input_shape = [cfg['PREPROCESS']['PARAMS']['WINDOW']] + cfg['PREPROCESS']['PARAMS']['IMG_SIZE'] + [3]
    model = model_def_fn(hparams, input_shape, metrics, n_classes)

    # Train and save the model
    epochs = cfg['TRAIN']['PARAMS']['EPOCHS']
    model.fit(train_set, epochs=epochs, validation_data=val_set, class_weight=class_weight, callbacks=[save_cp])

# Train and save the model
train_model()
