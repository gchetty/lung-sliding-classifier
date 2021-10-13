import cv2
import numpy as np
import pandas as pd
import os
import yaml
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops import random_ops
import tensorflow_addons as tfa

cfg = yaml.full_load(open(os.getcwd() + '/config.yml', 'r'))

class Preprocessor:

    def __init__(self, preprocess_fn):
        self.batch_size = cfg['TRAIN']['PARAMS']['BATCH_SIZE']
        self.autotune = tf.data.AUTOTUNE
        self.input_scaler = preprocess_fn

    def prepare(self, ds, shuffle=False, augment=False):

        ds = ds.map(self._parse_fn, num_parallel_calls=self.autotune)

        if shuffle:
            ds = ds.shuffle(1000)  # val should be dataset size - NEEDS ADJUSTMENT?

        ds = ds.batch(self.batch_size)

        if augment:
            ds = ds.map(lambda x, y: (augment(x), y), num_parallel_calls=self.autotune)

        return ds.prefetch(buffer_size=self.autotune)

    def _parse_fn(self, filename, label):
        clip = np.load(filename, allow_pickle=True)['frames']
        return clip, tf.one_hot(label, 2)  # hardcoded as binary, can change

    def augment(x):
        x = tf.cast(x, tf.float32)  # idk if needed
        x = tf.divide(x, tf.constant(255.0))  # Normalize
        x = tf.map_fn(lambda x1: tf.image.random_brightness(x1, max_delta=0.2), x)  # delta might need tuning
        x = tf.map_fn(lambda x1: tf.image.random_hue(x1, max_delta=0.5), x)  # delta might need tuning
        x = tf.map_fn(lambda x1: tf.image.random_contrast(x1, 0.5, 0.9), x)
        x = tf.map_fn(lambda x1: random_flip_left_right_clip(x1), x)
        x = tf.map_fn(lambda x1: random_flip_up_down_clip(x1), x)
        x = tf.map_fn(lambda x1: random_rotate_clip(x1), x)
        return x

    def random_flip_left_right_clip(x):
        # print(x.shape)
        r = random_ops.random_uniform_int([1], 0, 2)
        if tf.math.equal(tf.constant([1]), r):
            x = tf.image.flip_left_right(x)
        return x

    def random_flip_up_down_clip(x):
        r = random_ops.random_uniform_int([1], 0, 2)
        if tf.math.equal(tf.constant([1]), r):
            x = tf.image.flip_up_down(x)
        return x

    def random_rotate_clip(x):
        angle = random_ops.random_uniform([], -1.57, 1.57)
        x = tfa.image.rotate(x, angle)
        return x