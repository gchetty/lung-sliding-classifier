'''
Script for training experiments, including model training, hyperparameter search, cross validation, etc
'''

import yaml
import os
import datetime
import pandas as pd
import tensorflow as tf

from tensorflow.keras.metrics import Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives, Accuracy
from tensorflow_addons.metrics import F1Score, FBetaScore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocessor import Preprocessor, FlowPreprocessor
from visualization.visualization import log_confusion_matrix
from models.models import *
from custom.metrics import Specificity
from data.utils import refresh_folder


cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def train_model(model_def_str=cfg['TRAIN']['MODEL_DEF'], 
                hparams=cfg['TRAIN']['PARAMS']['INFLATED_RESNET50'],  # SHOULD REALLY MAKE THIS MORE DYNAMIC
                model_out_dir=cfg['TRAIN']['PATHS']['MODEL_OUT']):

    '''
    Trains and saves a model given specific hyperparameters

    :param model_def_str: A string in {'test1', ... } specifying the name of the desired model
    :param hparams: A dictionary specifying the hyperparameters to use
    :param model_out_dir: The path to save the model
    '''

    # Enable mixed precision
    mixed_precision = cfg['TRAIN']['MIXED_PRECISION']
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    flow = cfg['PREPROCESS']['PARAMS']['FLOW']
    
    # Get the model function and preprocessing function 
    model_def_fn, preprocessing_fn = get_model(model_def_str)

    CSVS_FOLDER = cfg['TRAIN']['PATHS']['CSVS']

    train_df = None
    val_df = None
    test_df = None

    train_set = None
    val_set = None
    test_set = None

    if not (flow == 'Yes'):

        # Read in training, validation, and test dataframes
        train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'train.csv'))#[:20]
        val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'val.csv'))#[:20]
        test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'test.csv'))#[:20]

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

    if not (flow == 'No'):

        train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'flow_train.csv'))  # [:20]
        val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'flow_val.csv'))  # [:20]
        test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'flow_test.csv'))  # [:20]

        # Create TF datasets for training, validation and test sets
        # Note: This does NOT load the dataset into memory! We specify paths,
        #       and labels so that TensorFlow can perform dynamic loading.
        train_set = tf.data.Dataset.from_tensor_slices((train_df['filename'].tolist(), train_df['label']))
        val_set = tf.data.Dataset.from_tensor_slices((val_df['filename'].tolist(), val_df['label']))
        test_set = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label']))

        # Create preprocessing object given the preprocessing function for model_def
        preprocessor = FlowPreprocessor(preprocessing_fn)

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

    counts = [num_no_sliding, num_sliding]

    # Defining Binary Classification Metrics
    metrics = ['accuracy', AUC(name='auc'), FBetaScore(num_classes=2, average='micro', threshold=0.5)]
    metrics += [Precision(), Recall()]
    metrics += [TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives(), Specificity()]

    # Get the model
    input_shape = None
    if flow == 'Yes':
        input_shape = [cfg['PREPROCESS']['PARAMS']['WINDOW']] + cfg['PREPROCESS']['PARAMS']['IMG_SIZE'] + [2]
    elif flow == 'No':
        input_shape = [cfg['PREPROCESS']['PARAMS']['WINDOW']] + cfg['PREPROCESS']['PARAMS']['IMG_SIZE'] + [3]

    model = model_def_fn(hparams, input_shape, metrics, counts)

    # Refresh the TensorBoard directory
    tensorboard_path = cfg['TRAIN']['PATHS']['TENSORBOARD']
    refresh_folder(tensorboard_path)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the Confusion Matrix Callback
    def log_confusion_matrix_wrapper(epoch, logs, model=model, val_df=val_df, val_dataset=val_set):
        return log_confusion_matrix(epoch, logs, model, val_df, val_dataset)

    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix_wrapper)

    # Log metrics
    log_dir = cfg['TRAIN']['PATHS']['TENSORBOARD'] + time
    basic_call = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Learning rate scheduler & logging LR
    writer1 = tf.summary.create_file_writer(log_dir + '/train')

    def scheduler(epoch, lr):
        '''
        Returns learning rate for the upcoming epoch based on a set schedule
        Decreases learning rate by a factor of e^-(DECAY_VAL) starting at epoch 15

        :param epoch: Integer, training epoch number
        :param lr: Float, current learning rate

        :return: Float, new learning rate
        '''
        learning_rate = lr
        if epoch > 50:
            learning_rate = lr * tf.math.exp(-1 * hparams['LR_DECAY_VAL'])
        with writer1.as_default():  # Write LR scalar to log directory
            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Creating a ModelCheckpoint for saving the model
    save_cp = ModelCheckpoint(model_out_dir, save_best_only=cfg['TRAIN']['SAVE_BEST_ONLY'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min',
                                   restore_best_weights=True)

    # Train and save the model
    epochs = cfg['TRAIN']['PARAMS']['EPOCHS']
    model.fit(train_set, epochs=epochs, validation_data=val_set, class_weight=class_weight,
              callbacks=[save_cp, cm_callback, basic_call, early_stopping, lr_callback], verbose=2)


# Train and save the model
train_model()
