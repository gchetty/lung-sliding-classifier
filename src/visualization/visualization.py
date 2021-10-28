'''
Script for generating Various plots and graphs
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools
import yaml
import io
import os

from sklearn.metrics import confusion_matrix

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
