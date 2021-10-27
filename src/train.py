'''
Script for training experiments, including model training, hyperparameter search, cross validation, etc
'''

from tensorflow.python.keras.metrics import TrueNegatives, TruePositives
import yaml
import os
import pandas as pd
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocessor import Preprocessor
from models.models import *
from data.utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    :param cm: (array, shape = [n, n]): a confusion matrix of integer classes
    :param class_names: (array, shape = [n]): String names of the integer classes

    Returns: The Matplotlib figure
    """
    
    figure = plt.figure(figsize=(16, 16))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    #cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot to a PNG image and returns it.

    :param figure: The Matplotlib figure
    
    Returns: The PNG image
    """
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def log_confusion_matrix(epoch, logs, model, val_df, val_dataset):
    '''
    Logs a Confusion Matrix image to TensorBoard.

    :param epoch: The epoch
    :param logs: The Keras fit logs
    :param model: The Keras model
    :param val_df: The dataframe containing the validation partition
    :param val_dataset: The TensorFlow dataset containing the validation partition
    '''
    paths, labels = val_df['filename'], val_df['label']
    test_pred_raw = model.predict(val_dataset).flatten()
    test_pred = (test_pred_raw > 0.5).astype(np.int)
    cm = confusion_matrix(labels, test_pred)
    figure = plot_confusion_matrix(cm, class_names=['No Sliding', 'Sliding'])
    cm_image = plot_to_image(figure)
    path = cfg['TRAIN']['PATHS']['TENSORBOARD']
    file_writer_cm = tf.summary.create_file_writer(path + 'temp/cm')

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


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
    train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'train.csv'))[:20]
    val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'val.csv'))[:25]
    test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'test.csv'))[:25]

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

    # Defining Binary Classification Metrics
    metrics = ['accuracy', AUC(name='auc'), F1Score(num_classes=1, threshold=0.5)]
    metrics += [Precision(), Recall()]
    metrics += [TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()]

    # Creating a ModelCheckpoint for saving the model
    save_cp = ModelCheckpoint(model_out_dir, save_best_only=cfg['TRAIN']['SAVE_BEST_ONLY'])

    # Get the model
    input_shape = [cfg['PREPROCESS']['PARAMS']['WINDOW']] + cfg['PREPROCESS']['PARAMS']['IMG_SIZE'] + [3]
    model = model_def_fn(hparams, input_shape, metrics)

    # Refresh the TensorBoard directory
    tensorboard_path = cfg['TRAIN']['PATHS']['TENSORBOARD']
    refresh_folder(tensorboard_path)

    # Create the Confusion Matrix Callback
    def log_confusion_matrix_wrapper(epoch, logs, model=model, val_df=val_df, val_dataset=val_set):
        return log_confusion_matrix(epoch, logs, model, val_df, val_dataset)

    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix_wrapper)

    # Train and save the model
    epochs = cfg['TRAIN']['PARAMS']['EPOCHS']
    model.fit(train_set, epochs=epochs, validation_data=val_set, class_weight=class_weight, callbacks=[save_cp, cm_callback])

# Train and save the model
train_model()
