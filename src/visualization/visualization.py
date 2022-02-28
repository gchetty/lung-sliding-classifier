'''
Script for generating Various plots and graphs
'''

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
import io
import os
import datetime
from sklearn.metrics import confusion_matrix, roc_curve
from skopt.plots import plot_objective

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))

def plot_roc(labels, predictions, class_name_list, dir_path=None, title=None):
    '''
    Plots the ROC curve for predictions on a dataset
    :param labels: Ground truth labels
    :param predictions: Model predictions corresponding to the labels
    :param class_name_list: Ordered list of class names
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    fp, tp, _ = roc_curve(labels, predictions)  # Get values for true positive and true negative
    plt.plot(100*fp, 100*tp, linewidth=2)   # Plot the ROC curve

    if title is None:
        plt.title('ROC curves for test set')
    else:
        plt.title(title)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-5,105])
    plt.ylim([-5,105])
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')
    if dir_path is not None:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(dir_path + 'ROC_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return plt

def plot_to_tensor():
    '''
    Converts a matplotlib figure to an image tensor
    :param figure: A matplotlib figure
    :return: Tensorflow tensor representing the matplotlib image
    '''
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image_tensor = tf.image.decode_png(buf.getvalue(), channels=4)     # Convert .png buffer to tensorflow image
    image_tensor = tf.expand_dims(image_tensor, 0)     # Add the batch dimension
    return image_tensor


def plot_bayesian_hparam_opt(model_name, hparam_names, search_results, save_fig=False):
    '''
    Plot all 2D hyperparameter comparisons from the logs of a Bayesian hyperparameter optimization.
    :param model_name: Name of the model
    :param hparam_names: List of hyperparameter identifiers
    :param search_results: The object resulting from a Bayesian hyperparameter optimization with the skopt package
    :param save_fig:
    :return:
    '''

    # Abbreviate hyperparameters to improve plot readability
    axis_labels = hparam_names.copy()
    for i in range(len(axis_labels)):
        if len(axis_labels[i]) >= 12:
            axis_labels[i] = axis_labels[i][:4] + '...' + axis_labels[i][-4:]

    # Plot
    axes = plot_objective(result=search_results, dimensions=axis_labels)

    # Create a title
    fig = plt.gcf()
    fig.suptitle('Bayesian Hyperparameter\n Optimization for ' + model_name, fontsize=15, x=0.65, y=0.97)

    # Indicate which hyperparameter abbreviations correspond with which hyperparameter
    hparam_abbrs_text = ''
    for i in range(len(hparam_names)):
        hparam_abbrs_text += axis_labels[i] + ':\n'
    fig.text(0.50, 0.8, hparam_abbrs_text, fontsize=10, style='italic', color='mediumblue')
    hparam_names_text = ''
    for i in range(len(hparam_names)):
        hparam_names_text += hparam_names[i] + '\n'
    fig.text(0.65, 0.8, hparam_names_text, fontsize=10, color='darkblue')

    fig.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(cfg['HPARAM_SEARCH']['PATH'], 'Bayesian_opt_' + model_name + '_' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'))


def plot_confusion_matrix(labels, predictions, class_name_list, dir_path=None, title=None):
    '''
    Plot a confusion matrix for the ground truth labels and corresponding model predictions for a particular class.
    :param labels: Ground truth labels
    :param predictions: Model predictions
    :param class_name_list: Ordered list of class names
    :param dir_path: Directory in which to save image
    :param title: Title of plot
    '''
    plt.clf()
    predictions = np.round(predictions)
    ax = plt.subplot()
    cm = confusion_matrix(list(labels), predictions)  # Determine confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Plot confusion matrix
    ax.figure.colorbar(im, ax=ax)
    ax.set(yticklabels=class_name_list, xticklabels=class_name_list)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=1, offset=0.5))
    ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(base=1, offset=0.5))

    # Print the confusion matrix numbers in the center of each cell of the plot
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # Set plot's title and axis names
    if title is None:
        plt.title('Confusion matrix for test set')
    else:
        plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # Save the image
    if dir_path is not None:
        plt.savefig(dir_path + 'CM_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    print('Confusion matrix: ', cm)    # Print the confusion matrix
    return plt


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
    file_writer_cm = tf.summary.create_file_writer(path + '/cm')

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def visualize_heatmap(orig_img, heatmap, img_filename, label, probs, class_names, dir_path=None):

    '''
    Obtain a comparison of an original image and heatmap produced by Grad-CAM.

    :param orig_img: Original LUS image
    :param heatmap: Heatmap generated by Grad-CAM.
    :param img_filename: Filename of the image explained
    :param label: Ground truth class of the example
    :param probs: Prediction probabilities
    :param class_names: Ordered list of class names
    :param dir_path: Path to save the generated image

    :return: Path to saved image
    '''

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(orig_img)
    ax[1].imshow(heatmap)

    # Display some information about the example
    pred_class = int(probs > 0.5)
    fig.text(0.02, 0.90, "Prediction probabilities for: " + str(class_names) + ': ' +
             str(['{:.2f}'.format(probs)]), fontsize=8)
    fig.text(0.02, 0.92, "Predicted Class: " + str(pred_class) + ' (' + class_names[pred_class] + ')', fontsize=8)
    if label is not None:
        fig.text(0.02, 0.94, "Ground Truth Class: " + str(label) + ' (' + class_names[label] + ')', fontsize=8)
    fig.suptitle(img_filename, fontsize=6, fontweight='bold')
    # fig.tight_layout()

    # Save the image
    filename = None
    if dir_path is not None:
        filename = os.path.join(dir_path, img_filename.split('/')[-1] + '_gradcam_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
        plt.savefig(filename)
    return filename
