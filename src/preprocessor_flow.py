import numpy as np
import os
import yaml
import tensorflow as tf
import preprocessor

cfg = yaml.full_load(open(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'config.yml')), 'r'))


def augment_flow(x):
    '''
    Augmentations for optical flow clips

    :param x:

    :return:
    '''

    x = tf.map_fn(
        lambda x1: tf.image.random_brightness(x1, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['BRIGHTNESS_DELTA']),
        x)
    x = tf.map_fn(lambda x1: tf.image.random_contrast(x1, cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][0],
                                                      cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][1]), x)
    x = tf.map_fn(lambda x1: preprocessor.random_shift_clip(x1), x)
    x = tf.map_fn(lambda x1: preprocessor.random_flip_left_right_clip(x1), x)
    x = tf.map_fn(lambda x1: preprocessor.random_zoom_clip(x1), x)
    return x


def augment_two_stream(x1, x2):
    '''
    Needed to ensure that augmentations are the same across two inputs

    :param x1: Mini-clip to augment
    :param x2: Associated flow mini-clip to augment

    :return: Tuple (x1, x2) of augmented mini-clip and flow mini-clip
    '''

    img_size = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    window = cfg['PREPROCESS']['PARAMS']['M_MODE_WIDTH']
    zero_pad = tf.constant(np.zeros([window] + img_size + [1]), dtype=tf.float32)
    x2 = tf.concat([x2, zero_pad], axis=-1)  # add third channel of 0s to flow clip

    x = tf.concat([x1, x2], axis=-4)  # Add mini-clip frames behind regular frames in time - making 1 sequence of frames

    # Call augmentation functions
    x = tf.map_fn(
        lambda x1: tf.image.random_brightness(x1, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['BRIGHTNESS_DELTA']),
        x)
    x = tf.map_fn(lambda x1: tf.image.random_contrast(x1, cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][0],
                                                      cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][1]), x)
    x = tf.map_fn(lambda x1: preprocessor.random_flip_left_right_clip(x1), x)
    x = tf.map_fn(lambda x1: preprocessor.random_noise(x1), x)

    # Separate regular and flow mini-clips
    x1, x2 = tf.split(x, 2, axis=-4)
    x2 = tf.split(x2, [2, 1], axis=-1)[0]
    x1.set_shape([window] + img_size + [3])
    x2.set_shape([window] + img_size + [2])

    return x1, x2


def parse_flow(filename, label):
    '''
    Loads a flow video (2 channels - x and y flow) from its filename and returns its label as proper tensors

    :param filename: Path to an .npz file
    :param label: Binary label for the video

    :return: Tuple of (Loaded Video, One-hot Tensor)
    '''

    img_size_tuple = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    num_frames = cfg['PREPROCESS']['PARAMS']['M_MODE_WIDTH']
    shape = (num_frames, img_size_tuple[0], img_size_tuple[1], 2)
    clip, label = tf.numpy_function(preprocessor.parse_fn, [filename, label], (tf.float32, tf.float32))
    clip.set_shape(shape)
    tf.ensure_shape(clip, shape)
    label.set_shape((1,))
    tf.ensure_shape(label, (1,))
    return clip, label


def parse_flow_only(filename):
    '''

    :param filename:
    :return:
    '''

    img_size_tuple = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    num_frames = cfg['PREPROCESS']['PARAMS']['M_MODE_WIDTH']
    shape = (num_frames, img_size_tuple[0], img_size_tuple[1], 2)
    clip = tf.numpy_function(preprocessor.parse_fn_no_label, [filename], tf.float32)
    tf.ensure_shape(clip, shape)
    return clip


class FlowPreprocessor:

    '''
    Preprocessor for datasets consisting of optical flow videos (2-channel) and their corresponding labels
    '''

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

        :return: A TF dataset with either preprocessed data or a full pipeline for eventual preprocessing
        '''

        # Shuffle the dataset
        if shuffle:
            shuffle_val = len(df)
            ds = ds.shuffle(shuffle_val)

        # Load the videos and create their labels as a one-hot vector
        ds = ds.map(parse_flow, num_parallel_calls=self.autotune)

        # Define batch size
        ds = ds.batch(self.batch_size, num_parallel_calls=self.autotune)

        # Median filter for smoothing
        ds = ds.map(lambda x, y: (preprocessor.median_filter(x), y), num_parallel_calls=self.autotune)

        # Optionally apply a series of augmentations
        if augment:
            ds = ds.map(lambda x, y: (augment_flow(x), y), num_parallel_calls=self.autotune)

        # Map the preprocessing (scaling, resizing) function to each element
        ds = ds.map(lambda x, y: (self.preprocessing_fn(x), y), num_parallel_calls=self.autotune)

        # Allows later elements to be prepared while the current element is being processed
        ds = ds.prefetch(buffer_size=self.autotune)

        return ds


class TwoStreamPreprocessor:

    '''
    Preprocessor for datasets consisting of LUS clips (or mini-clips) and corresponding optical flow videos (2-channel)
    as model input, and their corresponding labels
    '''

    def __init__(self, preprocessing_fns):
        self.batch_size = cfg['TRAIN']['PARAMS']['BATCH_SIZE']
        self.autotune = tf.data.AUTOTUNE
        self.preprocessing_fn_1 = preprocessing_fns[0]  # Preprocessing function for clips
        self.preprocessing_fn_2 = preprocessing_fns[1]  # Preprocessing function for flow

    def prepare(self, ds, df, shuffle=False, augment=False):

        '''
        Defines the pipeline for loading and preprocessing each item in a TF dataset

        :param ds: The TF dataset to define the pipeline for - must return items of the form ((clip, flow clip), label)
        :param df: The DataFrame corresponding to ds
        :param shuffle: A boolean to decide if shuffling is desired
        :param augment: A boolean to decide if augmentation is desired

        :return: A TF dataset with either preprocessed data or a full pipeline for eventual preprocessing
        '''

        # Shuffle the dataset
        if shuffle:
            shuffle_val = len(df)
            ds = ds.shuffle(shuffle_val)

        # Load clips and flow clips
        ds = ds.map(lambda x, y: ((preprocessor.parse_clip_only(x[0]), parse_flow_only(x[1])), y),
                    num_parallel_calls=self.autotune)

        # Optionally apply a series of augmentations
        if augment:
            ds = ds.map(lambda x, y: (augment_two_stream(x[0], x[1]), y), num_parallel_calls=self.autotune)

        # Define batch size
        ds = ds.batch(self.batch_size, num_parallel_calls=self.autotune)

        # Map the preprocessing (scaling, resizing) function to each element
        ds = ds.map(lambda x, y: ((self.preprocessing_fn_1(x[0]), self.preprocessing_fn_2(x[1])), y),
                    num_parallel_calls=self.autotune)

        # Allows later elements to be prepared while the current element is being processed
        ds = ds.prefetch(buffer_size=self.autotune)

        return ds

