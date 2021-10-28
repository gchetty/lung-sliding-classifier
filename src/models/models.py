'''
Script for defining TensorFlow neural network models
'''

import yaml
import os

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv3D, AveragePooling3D, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def get_model(model_name):
    '''
    Gets the function that returns the desired model function and its associated preprocessing function.

    :param model_name: A string in {'test1', ...} specifying the model

    :return: A Tuple (Function returning compiled model, Required preprocessing function)
    '''
    if model_name == 'lrcn':
        model_def_fn = lrcn
        preprocessing_fn = (lambda x: x / 255.0)
    elif model_name == 'threeDCNN':
        model_def_fn = threeDCNN
        preprocessing_fn = (lambda x: x / 255.0)
    return model_def_fn, preprocessing_fn


def lrcn(model_config, input_shape, metrics):
    """Build a CNN into RNN.
    Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
    Heavily influenced by VGG-16:
        https://arxiv.org/abs/1409.1556
    Also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """
    def add_default_block(model, kernel_filters, init, reg_lambda):
        # conv
        model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                         kernel_initializer=init)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # conv
        model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                         kernel_initializer=init)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # max pool
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        return model

    initialiser = 'glorot_uniform'
    reg_lambda = model_config['L2_LAMBDA']

    model = Sequential()

    # first (non-default) block
    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                     kernel_initializer=initialiser),
                              input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # 2nd-5th (default) blocks
    model = add_default_block(model, 64,  init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

    # LSTM output head
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=model_config['LR']), metrics=metrics)

    return model

def threeDCNN(model_config, input_shape, metrics):
    '''
    Returns a custom 3D CNN
    '''
    model = Sequential()

    model.add(Conv3D(filters=32, kernel_size=(2, 3, 3), strides=1, input_shape=input_shape))
    model.add(AveragePooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())
    
    model.add(Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1))
    model.add(AveragePooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=128, kernel_size=(2, 3, 3), strides=1))
    model.add(AveragePooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=256, kernel_size=(2, 3, 3), strides=1))
    model.add(AveragePooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=model_config['LR']), metrics=metrics)

    return model