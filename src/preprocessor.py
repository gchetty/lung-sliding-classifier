import numpy as np
import pandas as pd
import os
import yaml
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import random_ops
from copy import deepcopy

cfg = yaml.full_load(open(os.getcwd() + '/../config.yml', 'r'))


def random_flip_left_right_clip(x):
    '''
    Applies a left-right augmentation to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        x = tf.image.flip_left_right(x)
    return x


def random_flip_up_down_clip(x):
    '''
    Applies an up-down augmentation to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        x = tf.image.flip_up_down(x)
    return x


def random_rotate_clip(x):
    '''
    Randomly rotates a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        angle = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['ROTATE_RANGE'][0],
                                          cfg['TRAIN']['PARAMS']['AUGMENTATION']['ROTATE_RANGE'][1])
        x = tfa.image.rotate(x, angle)
    return x


def random_shift_clip(x):
    '''
    Applies a random horizontal and vertical translation to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        h, w = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0], cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][1]  # (clip_length, h, w, channels)
        dx = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHIFT_LEFTRIGHT_BOUNDS'][0],
                                       cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHIFT_LEFTRIGHT_BOUNDS'][1]) * h
        dy = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHIFT_UPDOWN_BOUNDS'][0],
                                       cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHIFT_UPDOWN_BOUNDS'][1]) * w
        translations = [[dx, dy]] * cfg['PREPROCESS']['PARAMS']['WINDOW']
        x = tfa.image.translate(x, translations)
    return x


def random_shear_clip(x):
    '''
    Applies a random shear to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        level = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHEAR_RANGE'][0],
                                          cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHEAR_RANGE'][1])
        replace = tf.constant([0.0, 0.0, 0.0])
        x = tf.map_fn(lambda x1: tfa.image.shear_x(x1, level, replace), x)
        x = tf.map_fn(lambda x1: tfa.image.shear_y(x1, level, replace), x)
    return x


def random_zoom_clip(x):
    '''
    Applies a random OUTWARDS zoom to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    
    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        prop = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['ZOOM_RANGE'][0],
                                         cfg['TRAIN']['PARAMS']['AUGMENTATION']['ZOOM_RANGE'][1])
        h_orig = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0]
        w_orig = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][1]
        h_changed = int(prop * h_orig)
        w_changed = int(prop * w_orig)
        x = tf.map_fn(lambda x1: tf.image.pad_to_bounding_box(x1, int((h_changed-h_orig)/2), int((w_changed-w_orig)/2),
                                                              h_changed, w_changed), x)
        x = tf.map_fn(lambda x1: tf.image.resize(x1, (h_orig, w_orig)), x)
    return x


def random_noise(x):
    '''
    Adds Gaussian noise to an inputted clip, with distribution parameters as specified in config file

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        mean = cfg['TRAIN']['PARAMS']['AUGMENTATION']['RANDOM_NOISE'][0]
        std = cfg['TRAIN']['PARAMS']['AUGMENTATION']['RANDOM_NOISE'][1]
        noise = tf.random.normal(x.shape[:-1] + [1], mean, std)  # Create noise in 1 channel only
        noise = tf.image.grayscale_to_rgb(noise)  # Convert noise to 3 channel
        x = tf.math.add(x, noise)
        x = tf.clip_by_value(x, 0, 255)
    return x


def median_filter(x):
    kernel_size = cfg['TRAIN']['PARAMS']['AUGMENTATION']['MEDIAN_FILTER_SIZE']
    x = tf.map_fn(lambda x1: tfa.image.median_filter2d(x1, kernel_size), x)
    return x


def augment_clip(x):
    '''
    Applies a series of transformations to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    x = tf.map_fn(lambda x1: tf.image.random_brightness(x1, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['BRIGHTNESS_DELTA']), x)
    #x = tf.map_fn(lambda x1: tf.image.random_hue(x1, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['HUE_DELTA']), x)
    x = tf.map_fn(lambda x1: tf.image.random_contrast(x1, cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][0],
                                                      cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][1]), x)
    x = tf.map_fn(lambda x1: random_shift_clip(x1), x)
    x = tf.map_fn(lambda x1: random_flip_left_right_clip(x1), x)
    #x = tf.map_fn(lambda x1: random_flip_up_down_clip(x1), x)  # DO NOT USE - Unrealistic for LUS clips
    x = tf.map_fn(lambda x1: random_rotate_clip(x1), x)
    #x = tf.map_fn(lambda x1: random_shear_clip(x1), x)
    x = tf.map_fn(lambda x1: random_zoom_clip(x1), x)
    x = tf.map_fn(lambda x1: random_noise(x1), x)
    return x


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
    x = tf.map_fn(lambda x1: random_shift_clip(x1), x)
    x = tf.map_fn(lambda x1: random_flip_left_right_clip(x1), x)
    x = tf.map_fn(lambda x1: random_zoom_clip(x1), x)
    # AUGMENT RANDOM NOISE WONT WORK RN BECAUSE
    return x


def augment_two_stream(x1, x2):
    '''
    Needed to ensure that augmentations are the same across two inputs

    :param x1: Mini-clip to augment
    :param x2: Associated flow mini-clip to augment

    :return: Tuple (x1, x2) of augmented mini-clip and flow mini-clip
    '''

    img_size = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    window = cfg['PREPROCESS']['PARAMS']['WINDOW']
    zero_pad = tf.constant(np.zeros([window] + img_size + [1]), dtype=tf.float32)
    x2 = tf.concat([x2, zero_pad], axis=-1)  # add third channel of 0s to flow clip

    x = tf.concat([x1, x2], axis=-4)  # Add mini-clip frames behind regular frames in time - making 1 sequence of frames

    # Call augmentation functions
    x = tf.map_fn(
        lambda x1: tf.image.random_brightness(x1, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['BRIGHTNESS_DELTA']),
        x)
    x = tf.map_fn(lambda x1: tf.image.random_contrast(x1, cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][0],
                                                      cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][1]), x)
    #x = tf.map_fn(lambda x1: random_shift_clip(x1), x)
    x = tf.map_fn(lambda x1: random_flip_left_right_clip(x1), x)
    #x = tf.map_fn(lambda x1: random_zoom_clip(x1), x)
    x = tf.map_fn(lambda x1: random_noise(x1), x)

    # Separate regular and flow mini-clips
    x1, x2 = tf.split(x, 2, axis=-4)
    x2 = tf.split(x2, [2, 1], axis=-1)[0]

    #tf.ensure_shape(x1, [window] + img_size + [3])
    x1.set_shape([window] + img_size + [3])
    #tf.ensure_shape(x2, [window] + img_size + [2])
    x2.set_shape([window] + img_size + [2])

    return x1, x2


def parse_fn(filename, label):
    '''
    Loads a video from its filename and returns its label

    :param filename: Path to an .npz file
    :param label: Binary label for the video

    :return: Tuple of (Loaded Video, One-hot Tensor)
    '''

    with np.load(filename, allow_pickle=True) as loaded_file:
        clip = loaded_file['frames']
    clip = tf.cast(clip, tf.float32)
    return clip, (1 - tf.one_hot(label, 1))


def parse_tf(filename, label):
    '''
    Loads a video from its filename and returns its label as proper tensors

    :param filename: Path to an .npz file
    :param label: Binary label for the video

    :return: Tuple of (Loaded Video, One-hot Tensor)
    '''

    img_size_tuple = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    num_frames = cfg['PREPROCESS']['PARAMS']['WINDOW']
    shape = (num_frames, img_size_tuple[0], img_size_tuple[1], 3)
    clip, label = tf.numpy_function(parse_fn, [filename, label], (tf.float32, tf.float32))
    clip.set_shape(shape)
    tf.ensure_shape(clip, shape)
    label.set_shape((1,))
    tf.ensure_shape(label, (1,))
    return clip, label


def parse_flow(filename, label):
    '''
    Loads a flow video (2 channels - x and y flow) from its filename and returns its label as proper tensors

    :param filename: Path to an .npz file
    :param label: Binary label for the video

    :return: Tuple of (Loaded Video, One-hot Tensor)
    '''

    img_size_tuple = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    num_frames = cfg['PREPROCESS']['PARAMS']['WINDOW']
    shape = (num_frames, img_size_tuple[0], img_size_tuple[1], 2)
    clip, label = tf.numpy_function(parse_fn, [filename, label], (tf.float32, tf.float32))
    clip.set_shape(shape)
    tf.ensure_shape(clip, shape)
    label.set_shape((1,))
    tf.ensure_shape(label, (1,))
    return clip, label


def parse_fn_no_label(filename):
    '''

    :param filename:
    :return:
    '''

    with np.load(filename, allow_pickle=True) as loaded_file:
        clip = loaded_file['frames']
    clip = tf.cast(clip, tf.float32)
    return clip


def parse_clip_only(filename):
    '''

    :param filename:
    :return:
    '''

    img_size_tuple = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    num_frames = cfg['PREPROCESS']['PARAMS']['WINDOW']
    shape = (num_frames, img_size_tuple[0], img_size_tuple[1], 3)
    clip = tf.numpy_function(parse_fn_no_label, [filename], tf.float32)
    tf.ensure_shape(clip, shape)
    return clip


def parse_flow_only(filename):
    '''

    :param filename:
    :return:
    '''

    img_size_tuple = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    num_frames = cfg['PREPROCESS']['PARAMS']['WINDOW']
    shape = (num_frames, img_size_tuple[0], img_size_tuple[1], 2)
    clip = tf.numpy_function(parse_fn_no_label, [filename], tf.float32)
    tf.ensure_shape(clip, shape)
    return clip


def parse_label(label):
    '''

    :param label:
    :return:
    '''

    label = tf.constant(label)
    label.set_shape((1,))
    tf.ensure_shape(label, (1,))
    return label


def augment_m_mode(x):
    '''
    Augments and M-mode image

    :param x: M-mode image to be augmented

    :return: Augmented M-mode image
    '''

    x = tf.image.random_brightness(x, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['BRIGHTNESS_DELTA'])
    x = tf.image.random_contrast(x, cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][0],
                                 cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][1])
    x = random_flip_left_right_clip(x)  # Function says clip but it works for single images as well
    return x


def augment_clip_m_mode_video(x):
    '''
    Augments a LUS clip (or mini-clip) before m-mode reconstruction

    :param x: LUS clip to be augmented

    :return: Augmented LUS clip
    '''
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.image.random_brightness(x, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['BRIGHTNESS_DELTA'])
    x = tf.image.random_contrast(x, cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][0],
                                 cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][1])
    x = random_flip_left_right_clip(x)  # Function says clip but these work for single images as well
    x = random_shift_clip(x)
    x = random_rotate_clip(x)
    return x


def parse_tf_m_mode(filename, label, augment=False):
    '''
    Loads an LUS clip corresponding to an inputted filename, and returns the corresponding M-mode reconstruction
    as well as the associated label

    :param filename: Path to LUS clip to load
    :param label: Binary label indicating whether the LUS clip exhibits lung sliding
    :param augment: Whether to augment LUS clips before extracting M-mode images

    :return: Tuple of (M-mode reconstruction, label), both as tensors
    '''

    img_height = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0]
    num_frames = cfg['PREPROCESS']['PARAMS']['WINDOW']
    shape = (img_height, num_frames, 3)
    pic, label = tf.numpy_function(parse_fn_m_mode, [filename, label, augment], (tf.float32, tf.float32))
    pic.set_shape(shape)
    label.set_shape((1,))
    return pic, label


def parse_fn_m_mode(filename, label, augment=False):
    '''
    Loads an LUS clip, optionally augments it, and extracts an M-mode representation

    :param filename: Path to LUS clip to load
    :param label: Binary label indicating whether the LUS clip exhibits lung sliding
    :param augment: Whether to augment LUS clips before extracting M-mode images

    :return: Tuple of (M-mode reconstruction, label), both as tensors
    '''
    loaded = np.load(filename)
    bounding_box = None
    if cfg['PREPROCESS']['PARAMS']['USE_BOUNDING_BOX']:
        clip, bounding_box, height_width = loaded['frames'], loaded['bounding_box'], loaded['height_width']
    else:
        clip, height_width = loaded['frames'], loaded['height_width']

    # Augment clip before extracting m-mode
    if augment:
        clip = augment_clip_m_mode_video(clip)
    clip = tf.clip_by_value(clip, 0, 255)

    # Extract m-mode
    num_frames, new_height, new_width = clip.shape[0], clip.shape[1], clip.shape[2]
    method = cfg['TRAIN']['M_MODE_SLICE_METHOD']
    middle_pixel = get_middle_pixel_index(clip[0], bounding_box, height_width, method=method)

    # Fix bad bounding box
    if middle_pixel == 0:
        middle_pixel = new_width // 2

    three_slice = clip[:, :, middle_pixel - 1:middle_pixel + 1, 0]
    mmode = np.median(three_slice, axis=2).T
    mmode_image = mmode.reshape((new_height, num_frames, 1))
    as_tensor = tf.convert_to_tensor(mmode_image)
    converted = tf.image.grayscale_to_rgb(as_tensor)
    final_pic = tf.cast(converted, tf.float32)
    return final_pic, (1 - tf.one_hot(label, 1))


def get_middle_pixel_index(image, bounding_box, original_height_width, method='brightest', apply_median_filter=True):

    image = deepcopy(image)

    if apply_median_filter:
        image = tfa.image.median_filter2d(image, filter_shape=(3, 3))

    middle_pixel_index = None

    if method == 'random':
        method = np.random.choice(['brightest', 'box_middle', 'brightest_vertical_sum'])

    if method == 'brightest':
        middle_pixel_index = get_middle_pixel_index_brightest(image, bounding_box, original_height_width)

    elif method == 'box_middle':
        middle_pixel_index = get_middle_pixel_index_box_middle(image, bounding_box, original_height_width)

    elif method == 'brightest_vertical_sum':
        middle_pixel_index = get_middle_pixel_index_brightest_vertical_sum(image, bounding_box, original_height_width)

    return middle_pixel_index


def get_middle_pixel_index_brightest(image, bounding_box, original_height_width):
    max_horizontal_slice = np.max(image, axis=0)
    middle_pixel_index = np.argmax(max_horizontal_slice)
    return middle_pixel_index


def get_middle_pixel_index_box_middle(image, bounding_box, original_height_width):
    resized_width = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][1]
    o_height, o_width = original_height_width
    ymin, xmin, ymax, xmax = bounding_box
    original_middle = (xmin + xmax) / 2
    width_frac = original_middle / o_width
    return int(resized_width * (width_frac))


def get_middle_pixel_index_brightest_vertical_sum(image, bounding_box, original_height_width):
    sum_horizontal_slice = np.sum(image, axis=0)
    middle_pixel_index = np.argmax(sum_horizontal_slice)
    return middle_pixel_index

# PREPROCESSOR CLASS THAT HANDLES NO LABELS - FOR HOLDOUTS???


class Preprocessor:

    '''
    Preprocessor for datasets consisting of LUS clips (or mini-clips) and their corresponding labels
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
        ds = ds.map(lambda x, y: (median_filter(x), y), num_parallel_calls=self.autotune)

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
        ds = ds.map(lambda x, y: ((parse_clip_only(x[0]), parse_flow_only(x[1])), y),
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


class MModePreprocessor:

    '''
    Preprocessor for datasets consisting of LUS clips (or mini-clips) and their corresponding labels
    From which m-mode reconstructions will be assembled and used as inputs
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

        # Augment clip and extract m-mode reconstructions
        ds = ds.map(lambda x, y: parse_tf_m_mode(x, y, augment), num_parallel_calls=self.autotune)

        # Define batch size
        ds = ds.batch(self.batch_size, num_parallel_calls=self.autotune)

        # Optionally augment m-mode images
        if augment:
            ds = ds.map(lambda x, y: (augment_m_mode(x), y), num_parallel_calls=self.autotune)

        # Map the preprocessing (scaling, resizing) function to each element
        ds = ds.map(lambda x, y: (self.preprocessing_fn(x), y), num_parallel_calls=self.autotune)

        # Allows later elements to be prepared while the current element is being processed
        ds = ds.prefetch(buffer_size=self.autotune)

        return ds
