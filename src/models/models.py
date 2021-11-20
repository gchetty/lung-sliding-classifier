'''
Script for defining TensorFlow neural network models
'''

import yaml
import os
import math
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv3D, AveragePooling3D, Dropout, Input, Add, InputLayer, MaxPooling3D,\
    Conv2D, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def get_model(model_name):
    '''
    Gets the function that returns the desired model function and its associated preprocessing function.

    :param model_name: A string in {'test1', ...} specifying the model

    :return: A Tuple (Function returning compiled model, Required preprocessing function)
    '''

    flow = True if cfg['PREPROCESS']['PARAMS']['FLOW'] == 'Yes' else False

    if model_name == 'lrcn':
        model_def_fn = lrcn
        preprocessing_fn = normalize
    elif model_name == 'threeDCNN':
        model_def_fn = threeDCNN
        preprocessing_fn = normalize
    elif model_name == 'xception_raw':
        model_def_fn = xception_raw
        preprocessing_fn = xception_preprocess
    elif model_name == 'vgg16_raw':
        model_def_fn = vgg16_raw
        preprocessing_fn = vgg16_preprocess
    elif model_name == 'res3d':
        model_def_fn = res3d
        preprocessing_fn = normalize
    elif model_name == 'inflated_resnet50':
        model_def_fn = inflated_resnet50
        preprocessing_fn = resnet_preprocess

    if flow:
        preprocessing_fn = normalize

    return model_def_fn, preprocessing_fn


def normalize(x):
    '''
    Converts input with pixel values [0, 255] to range [0, 1]

    :param x: n-dimensional image to be normalized
    :return: Normalized input
    '''
    return x / 255.0


def xception_raw(model_config, input_shape, metrics, class_counts):

    '''
    Time-distributed raw (not pre-trained) Xception feature extractor with LSTM and output head on top

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    optimizer = Adam(learning_rate=lr)

    base = Xception(include_top=False, weights=None, input_shape=input_shape[1:], pooling='avg')

    x_input = Input(input_shape, name='input')

    # First block
    x = TimeDistributed(base.layers[1], name=base.layers[1].name)(x_input)
    for i in range(2, 7):
        x = TimeDistributed(base.layers[i], name=base.layers[i].name)(x)
    prev_add = x

    # Second block
    for i in range(7, 12):
        x = TimeDistributed(base.layers[i], name=base.layers[i].name)(x)
    x = TimeDistributed(base.layers[13], name=base.layers[13].name)(x)  # batch norm
    x = TimeDistributed(base.layers[14], name=base.layers[14].name)(x)  # max pooling
    skip = TimeDistributed(base.layers[12], name=base.layers[12].name)(prev_add)  # Conv2d residual branch
    x = TimeDistributed(Add(), name=base.layers[15].name)([x, skip])
    prev_add = x

    # Third block
    for i in range(16, 22):
        x = TimeDistributed(base.layers[i], name=base.layers[i].name)(x)
    x = TimeDistributed(base.layers[23], name=base.layers[23].name)(x)
    x = TimeDistributed(base.layers[24], name=base.layers[24].name)(x)
    skip = TimeDistributed(base.layers[22], name=base.layers[22].name)(prev_add)  # Conv2d residual branch
    x = TimeDistributed(Add(), name=base.layers[25].name)([x, skip])
    prev_add = x

    # Fourth block
    for i in range(26, 32):
        x = TimeDistributed(base.layers[i], name=base.layers[i].name)(x)
    x = TimeDistributed(base.layers[33], name=base.layers[33].name)(x)
    x = TimeDistributed(base.layers[34], name=base.layers[34].name)(x)
    skip = TimeDistributed(base.layers[32], name=base.layers[32].name)(prev_add)  # Conv2d residual branch
    x = TimeDistributed(Add(), name=base.layers[35].name)([x, skip])
    prev_add = x

    # Blocks 5 through 12
    for n in range(0, 71, 10):
        for i in range(36 + n, 45 + n):
            x = TimeDistributed(base.layers[i], name=base.layers[i].name)(x)
        x = TimeDistributed(Add(), name=base.layers[45 + n].name)([x, prev_add])
        prev_add = x

    # Block 13
    for i in range(116, 122):
        x = TimeDistributed(base.layers[i], name=base.layers[i].name)(x)
    x = TimeDistributed(base.layers[123], name=base.layers[123].name)(x)
    x = TimeDistributed(base.layers[124], name=base.layers[124].name)(x)
    skip = TimeDistributed(base.layers[122], name=base.layers[122].name)(prev_add)  # Conv2d residual branch
    x = TimeDistributed(Add(), name=base.layers[125].name)([x, skip])

    # Block 14
    for layer in base.layers[126:]:
        x = TimeDistributed(layer, name=layer.name)(x)

    # LSTM and output head
    x = LSTM(256, return_sequences=False, dropout=dropout)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=x_input, outputs=outputs)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model


def vgg16_raw(model_config, input_shape, metrics, class_counts):

    '''
    Time-distributed raw (not pre-trained) VGG16 feature extractor with LSTM and output head on top

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    optimizer = Adam(learning_rate=lr)

    base = VGG16(include_top=False, weights=None, input_shape=input_shape[1:], pooling='avg')

    model = tf.keras.models.Sequential()

    model.add(InputLayer(input_shape=input_shape))

    for layer in base.layers[1:]:
        model.add(TimeDistributed(layer))

    model.add(LSTM(256, return_sequences=False, dropout=dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model


def lrcn(model_config, input_shape, metrics, class_counts):
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


def threeDCNN(model_config, input_shape, metrics, class_counts):

    '''
    Custom 3D CNN with 4 convolutional layers and output head

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    model = Sequential()

    model.add(Conv3D(filters=32, kernel_size=(2, 3, 3), strides=1, activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())
    
    model.add(Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=128, kernel_size=(2, 3, 3), strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=256, kernel_size=(2, 3, 3), strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling3D())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=model_config['LR']), metrics=metrics)

    return model


def res3d(model_config, input_shape, metrics, class_counts):

    '''
    Creates raw (not pre-trained) 3D CNN, based on ResNet-12 block structure,
    with 4 convolutional blocks and output head

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    optimizer = Adam(learning_rate=lr)

    output_bias = None
    if cfg['TRAIN']['OUTPUT_BIAS']:
        count0 = class_counts[0]
        count1 = class_counts[1]
        output_bias = math.log(count1/count0)
        output_bias = tf.keras.initializers.Constant(output_bias)

    inputs = Input(shape=input_shape)

    # block 1 (input)
    x = Conv3D(filters=32, kernel_size=(2, 3, 3), strides=1, activation='relu')(inputs)
    x = MaxPooling3D(pool_size=(2, 3, 3))(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1, activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 3, 3))(x)
    x = BatchNormalization()(x)
    prev = x

    # block 2 - conv, relu, BN, conv, add, relu, BN
    x = Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1, padding='same')(x)
    x = Add()([x, prev])
    x = Activation(activation='relu')(x)
    x = BatchNormalization()(x)
    prev = x

    # block 3 - conv, relu, BN, conv, add, relu, BN
    x = Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1, padding='same')(x)
    x = Add()([x, prev])
    x = Activation(activation='relu')(x)
    x = BatchNormalization()(x)
    prev = x

    # block 4 - conv, relu, BN, conv, add, relu, BN
    x = Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=64, kernel_size=(2, 3, 3), strides=1, padding='same')(x)
    x = Add()([x, prev])
    x = Activation(activation='relu')(x)
    x = BatchNormalization()(x)

    # block 5 (output)
    x = GlobalAveragePooling3D()(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid', bias_initializer=output_bias, dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model


def inflated_resnet50(model_config, input_shape, metrics, class_counts):

    '''
    Creates ResNet50 model inflated to 3 dimensions (as per Quo Vadis)
    Inflated iImagenet weights are used if inputs have 3 channels

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    flow = True if cfg['PREPROCESS']['PARAMS']['FLOW'] == 'Yes' else False

    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    optimizer = Adam(learning_rate=lr)

    output_bias = None
    if cfg['TRAIN']['OUTPUT_BIAS']:
        count0 = class_counts[0]
        count1 = class_counts[1]
        output_bias = math.log(count1 / count0)
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Download 2D ResNet50
    if flow:
        base = ResNet50V2(include_top=False, input_tensor=None, input_shape=(input_shape[1:3]+[3]), pooling='avg')
    else:
        base = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape[1:],
                          pooling='avg')

    def block(x, last, filters, ind, pos):

        '''
        Adds layers corresponding to a ResNet50 block (inflated to 3D)

        :param x: Output tensor from previous layers
        :param last: Last block's output tensor
        :param filters: Base number of filters in convolutions
        :param ind: Integer, tracking index of base ResNet50 layers (for naming purposes)
        :param pos: String, position of block in stage - either 'first', 'last', or a different arbitrary value

        :return: Tensor - block output
        '''

        x = BatchNormalization(name=base.layers[ind].name)(x)
        x = Activation(activation='relu', name=base.layers[ind + 1].name)(x)

        if pos == 'first':
            last = x

        x = Conv3D(filters=filters, kernel_size=1, strides=1, use_bias=False, name=base.layers[ind + 2].name)(x)
        x = BatchNormalization(name=base.layers[ind + 3].name)(x)
        x = Activation(activation='relu', name=base.layers[ind + 4].name)(x)
        x = ZeroPadding3D(padding=1, name=base.layers[ind + 5].name)(x)

        if pos == 'last':
            x = Conv3D(filters=filters, kernel_size=3, strides=2, use_bias=False, name=base.layers[ind + 6].name)(x)
        else:
            x = Conv3D(filters=filters, kernel_size=3, strides=1, use_bias=False, name=base.layers[ind + 6].name)(x)
        x = BatchNormalization(name=base.layers[ind + 7].name)(x)
        x = Activation(activation='relu', name=base.layers[ind + 8].name)(x)

        # ADD STUFF
        if pos == 'first':
            res = Conv3D(filters=filters * 4, kernel_size=1, strides=1, name=base.layers[ind + 9].name)(last)
            x = Conv3D(filters=filters * 4, kernel_size=1, strides=1, name=base.layers[ind + 10].name)(x)
            x = Add(name=base.layers[ind + 11].name)([x, res])
        elif pos == 'last':
            res = MaxPooling3D(pool_size=1, strides=2, name=base.layers[ind + 9].name)(last)
            x = Conv3D(filters=filters * 4, kernel_size=1, strides=1, name=base.layers[ind + 10].name)(x)
            x = Add(name=base.layers[ind + 11].name)([x, res])
        else:
            x = Conv3D(filters=filters * 4, kernel_size=1, strides=1, name=base.layers[ind + 9].name)(x)
            x = Add(name=base.layers[ind + 10].name)([x, last])

        return x

    def stage(x, last, num_blocks, filters, ind, final):

        '''
        Adds layers corresponding to a ResNet50 stage (inflated to 3D)

        :param x: Output tensor from previous layers
        :param last: Last block's output tensor
        :param num_blocks: Number of convolutional blocks in the stage
        :param filters: Base number of filters in convolutions
        :param ind: Integer, tracking index of base ResNet50 layers (for naming purposes)
        :param final: Boolean, flagging whether final stage or not (final stage has no dimensional reduction)

        :return: Tuple of (stage output tensor, index for next layer to be added)
        '''

        for i in range(num_blocks):
            pos = ''
            if i == 0:
                pos = 'first'
            elif i == (num_blocks - 1):
                if not final:
                    pos = 'last'
            x = block(x, last, filters, ind, pos)
            last = x
            if (pos == 'first') or (pos == 'last'):
                ind += 12
            else:
                ind += 11
            print(ind)
        return x, ind

    # USEFUL NOTE: Any conv with a BN directly after has no biases (handled by BN)

    # Input block
    inputs = tf.keras.Input(shape=input_shape)

    x = ZeroPadding3D(padding=3, name=base.layers[1].name)(inputs)
    x = Conv3D(filters=64, kernel_size=7, strides=2, name=base.layers[2].name)(x)
    x = ZeroPadding3D(padding=1, name=base.layers[3].name)(x)
    x = MaxPooling3D(pool_size=3, strides=2, name=base.layers[4].name)(x)

    index = 5

    x, index = stage(x, x, 3, 64, index, False)  # stage 1
    x, index = stage(x, x, 4, 128, index, False)  # stage 2
    x, index = stage(x, x, 6, 256, index, False)  # stage 3
    x, index = stage(x, x, 3, 512, index, True)  # stage 4

    # Output head
    x = BatchNormalization(name=base.layers[-3].name)(x)
    x = Activation(activation='relu', name=base.layers[-2].name)(x)
    x = GlobalAveragePooling3D(name=base.layers[-1].name)(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation='sigmoid', bias_initializer=output_bias, dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # Bootstrap weights & biases and freeze layers, only for convolutional and batch norm layers
    if not flow:

        for i in range(len(model.layers)):

            new_layer = model.layers[i]
            name = new_layer.name

            if '_conv' in name:
                base_layer = base.get_layer(name)
                orig_w = base_layer.get_weights()  # Params for corresponding 2D convolution from ResNet50
                expand = base_layer.kernel_size[0]  # Kernel size of layer - let this be 'expand'
                w = np.tile(orig_w[0], (expand, 1, 1, 1, 1))  # Duplicate corresponding 2D conv kernel 'expand' times
                w = np.divide(w, expand)  # Divide by expansion amount to keep filter response constant
                if len(orig_w) == 1:  # if no biases, only set weights
                    new_layer.set_weights([w])
                else:  # otherwise also take biases from corresponding 2D convolution layer
                    new_layer.set_weights([w, orig_w[1]])
            elif '_bn' in name:
                if not (name == 'post_bn'):  # input shape of post_bn is different if not using whole ResNet
                    new_layer.set_weights(base.get_layer(name).get_weights())

            if i < model_config['LAST_FROZEN']:
                new_layer.trainable = False

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model

