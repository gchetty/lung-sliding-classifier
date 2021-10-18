'''
Script for defining TensorFlow neural network models
'''

import yaml
import os

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def get_model(model_name):
    '''
    Gets the function that returns the desired model function and its associated preprocessing function
    :param model_name: A string in {'test1', ...} specifying the model
    returns: A Tuple (Function returning compiled model, Required preprocessing function)
    '''
    if model_name == 'test1':
        model_def_fn = test1
        preprocessing_fn = (lambda x: x / 255.0)

    return model_def_fn, preprocessing_fn


def test1(model_config, input_shape, metrics, n_classes):
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
    model.add(Dense(2, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=model_config['LR']), metrics=metrics)

    return model
