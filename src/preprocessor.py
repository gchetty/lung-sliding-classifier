import numpy as np
import pandas as pd
import os
import yaml
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import random_ops


cfg = yaml.full_load(open(os.getcwd() + '/../config.yml', 'r'))


def random_flip_left_right_clip(x):
    '''
    With probability 50%, applies a left-right augmentation to a video
    :param x: Tensor of shape (Clip_length, Height, Width, 3)
    returns: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        x = tf.image.flip_left_right(x)
    return x


def random_flip_up_down_clip(x):
    '''
    With probability 50%, applies an up-down augmentation to a video
    :param x: Tensor of shape (Clip_length, Height, Width, 3)
    returns: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        x = tf.image.flip_up_down(x)
    return x


def random_rotate_clip(x):
    '''
    With probability 50%, randomly rotates a video
    :param x: Tensor of shape (Clip_length, Height, Width, 3)
    returns: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        angle = random_ops.random_uniform([], -1.57, 1.57)
        x = tfa.image.rotate(x, angle)
    return x


def random_shift_clip(x):
    '''
    With probability 50%, applies random horizontal and vertical translation to a video
    :param x: Tensor of shape (Clip_length, Height, Width, 3)
    returns: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        h, w = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0], cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][1]  # (clip_length, h, w, channels)
        dx = random_ops.random_uniform([], -0.25, 0.25) * h
        dy = random_ops.random_uniform([], -0.0,
                                       0.2) * w  # No upwards shift since many areas of interest are near the top
        translations = [[dx, dy]] * cfg['PREPROCESS']['PARAMS']['WINDOW']
        x = tfa.image.translate(x, translations)
    return x


def random_shear_clip(x):
    '''
    With probability 50%, applies a random shear to a video
    :param x: Tensor of shape (Clip_length, Height, Width, 3)
    returns: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        level = random_ops.random_uniform([], 0.0, 0.25)  # may need adjustment
        replace = tf.constant([0.0, 0.0, 0.0])
        x = tf.map_fn(lambda x1: tfa.image.shear_x(x1, level, replace), x)
        x = tf.map_fn(lambda x1: tfa.image.shear_y(x1, level, replace), x)
    return x


def random_zoom_clip(x):
    '''
    With probability 50%, applies a random OUTWARDS zoom to a video
    :param x: Tensor of shape (Clip_length, Height, Width, 3)
    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        prop = random_ops.random_uniform([], 1.0, 1.5)  # tunable
        h_orig = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0]
        w_orig = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][1]
        h_changed = int(prop * h_orig)
        w_changed = int(prop * w_orig)
        x = tf.map_fn(lambda x1: tf.image.pad_to_bounding_box(x1, int((h_changed-h_orig)/2), int((w_changed-w_orig)/2), h_changed, w_changed), x)
        x = tf.map_fn(lambda x1: tf.image.resize(x1, (h_orig, w_orig)), x)
    return x


def augment_clip(x):
    '''
    Applies a series of transformations to a video
    :param x: Tensor of shape (Clip_length, Height, Width, 3)
    returns: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    x = tf.map_fn(lambda x1: tf.image.random_brightness(x1, max_delta=0.2), x)  # delta might need tuning
    x = tf.map_fn(lambda x1: tf.image.random_hue(x1, max_delta=0.5), x)  # delta might need tuning
    x = tf.map_fn(lambda x1: tf.image.random_contrast(x1, 0.5, 0.9), x)
    x = tf.map_fn(lambda x1: random_flip_left_right_clip(x1), x)
    x = tf.map_fn(lambda x1: random_flip_up_down_clip(x1), x)
    x = tf.map_fn(lambda x1: random_rotate_clip(x1), x)
    x = tf.map_fn(lambda x1: random_shear_clip(x1), x)
    x = tf.map_fn(lambda x1: random_zoom_clip(x1), x)
    x = tf.map_fn(lambda x1: random_shift_clip(x1), x)
    return x


def parse_fn(filename, label):
    '''
    Loads a video from its filename and returns its label
    :param filename: Path to an .npz file
    :param label: Binary label for the video
    returns: Tuple of (Loaded Video, One-hot Tensor)
    '''
    clip = np.load(filename, allow_pickle=True)['frames']
    clip = tf.cast(clip, tf.float32)
    return clip, tf.one_hot(label, 2)  # hardcoded as binary, can change


def parse_tf(filename, label):
    '''
    Loads a video from its filename and returns its label as proper tensors
    :param filename: Path to an .npz file
    :param label: Binary label for the video
    returns: Tuple of (Loaded Video, One-hot Tensor)
    '''
    shape = (cfg['PREPROCESS']['PARAMS']['WINDOW'], 128, 128, 3)
    clip, label = tf.numpy_function(parse_fn, [filename, label], (tf.float32, tf.float32))
    tf.ensure_shape(clip, shape)
    label.set_shape((2,))
    tf.ensure_shape(label, (2,))
    return clip, label


class Preprocessor:

    def __init__(self, preprocessing_fn):
        self.batch_size = cfg['TRAIN']['PARAMS']['BATCH_SIZE']
        self.autotune = tf.data.AUTOTUNE
        self.preprocessing_fn = preprocessing_fn

    def prepare(self, ds, df, shuffle=False, augment=False):
        '''
        Defines the pipeline for loading and preprocessing each item in a TF dataset
        :param ds: The TF dataset to define the pipeline for
        :param df: The DataFrame corresponding to ds
        :param shuffle: A boolean to decide if shuffling is desired
        :param augment: A boolean to decide if augmentation is desired
        returns ds: A TF dataset with either preprocessed data or a full pipeline for eventual preprocessing
        '''

        # Shuffle the dataset
        if shuffle:
            ds = ds.shuffle(len(df))

        # Load the videos and create their labels as a one-hot vector
        ds = ds.map(parse_tf, num_parallel_calls=self.autotune)
        
        # Define batch size
        ds = ds.batch(self.batch_size, num_parallel_calls=self.autotune)

        # Optionally apply a series of augmentations
        if augment:
            ds = ds.map(lambda x, y: (augment_clip(x), y), num_parallel_calls=self.autotune)
        
        # Map the preprocessing (scaling, resizing) function to each element
        ds = ds.map(lambda x, y: (self.preprocessing_fn(x), y), num_parallel_calls=self.autotune)

        # Allows later elements to be prepared while the current element is being processed
        ds = ds.prefetch(buffer_size=self.autotune)

        return ds







