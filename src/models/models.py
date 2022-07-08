'''
Script for defining TensorFlow neural network models
'''
import keras.legacy_tf_layers.variable_scope_shim
from tensorflow.keras import backend as K
import yaml
import os
import math
import tempfile
import tensorflow as tf
import numpy as np
import math

from tensorflow.keras.layers import Dense, Flatten, Lambda, Reshape, Conv1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv3D, AveragePooling3D, Dropout, Input, Add, InputLayer, \
    MaxPooling3D, \
    Conv2D, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling3D, ZeroPadding3D, Concatenate, \
    GlobalAveragePooling2D, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding, Resizing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2, L1
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB0 as EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import tensorflow_addons as tfa

cfg = yaml.full_load(open(os.path.join(os.getcwd(), 'config.yml'), 'r'))


def get_model(model_name):
    '''
    Gets the function that returns the desired model function and its associated preprocessing function.

    :param model_name: A string in the list below specifying the model

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
    elif model_name == 'vgg16':
        model_def_fn = vgg16
        preprocessing_fn = vgg16_preprocess
    elif model_name == 'res3d':
        model_def_fn = res3d
        preprocessing_fn = normalize
    elif model_name == 'inflated_resnet50':
        model_def_fn = inflated_resnet50
        preprocessing_fn = resnet_preprocess
    elif model_name == 'i3d':
        model_def_fn = i3d
        preprocessing_fn = [normalize, normalize]
    elif model_name == 'xception':
        model_def_fn = xception
        preprocessing_fn = xception_preprocess
    elif model_name == 'vit':
        model_def_fn = vit
        preprocessing_fn = normalize
    elif model_name == 'variance_mmode_net':
        model_def_fn = variance_mmode_net
        preprocessing_fn = normalize
    elif model_name == 'one_d_conv':
        model_def_fn = one_d_conv
        preprocessing_fn = normalize
    elif model_name == 'mobilenet_v3_small':
        model_def_fn = mobile_net_v3_small
        preprocessing_fn = mobilenet_v3_preprocess
    elif model_name == 'efficientnet':
        model_def_fn = efficientnet
        preprocessing_fn = efficientnet_preprocess

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


def vgg16(model_config, input_shape, metrics, class_counts):
    '''
    Time-distributed raw (not pre-trained) VGG16 feature extractor with LSTM and output head on top

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    flow = True if cfg['PREPROCESS']['PARAMS']['FLOW'] == 'Yes' else False

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)

    dropout = model_config['DROPOUT']
    l2_reg = model_config['L2_REG']  # NOT USED RN

    output_bias = None
    if cfg['TRAIN']['OUTPUT_BIAS']:
        count0 = class_counts[0]
        count1 = class_counts[1]
        output_bias = math.log(count1 / count0)
        output_bias = tf.keras.initializers.Constant(output_bias)

    kernel_init = model_config['WEIGHT_INITIALIZER']  # NOT USED IN CONV LAYERS RN

    # Download VGG16
    transfer = model_config['TRANSFER']
    if flow or (not transfer):
        base = VGG16(include_top=False, input_tensor=None, input_shape=(input_shape[1:3] + [3]), pooling='avg')
    else:
        base = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape[1:],
                     pooling='avg')

    model = tf.keras.models.Sequential()

    model.add(InputLayer(input_shape=input_shape))

    stop_block = model_config['BLOCKS'] + 1
    for layer in base.layers[1:-1]:
        if int(layer.name[5:6]) < stop_block:  # only add blocks up to specified 'last block'
            model.add(TimeDistributed(layer, name=layer.name))

    model.add(TimeDistributed(base.layers[-1], name=base.layers[-1].name))  # GAP
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=kernel_init, bias_initializer=output_bias,
                    dtype='float32'))

    model.summary()

    # Freeze layers
    for i in range(len(model.layers)):
        if i < model_config['LAST_FROZEN']:
            model.layers[i].trainable = False

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
    model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer=initialiser)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # 2nd-5th (default) blocks
    model = add_default_block(model, 64, init=initialiser, reg_lambda=reg_lambda)
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
    # model.add(Dropout(0.2))

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
        output_bias = math.log(count1 / count0)
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
    Inflated Imagenet weights are used if inputs have 3 channels

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    flow = True if cfg['PREPROCESS']['PARAMS']['FLOW'] == 'Yes' else False

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)

    dropout = model_config['DROPOUT']
    l2_reg = model_config['L2_REG']

    output_bias = None
    if cfg['TRAIN']['OUTPUT_BIAS']:
        count0 = class_counts[0]
        count1 = class_counts[1]
        output_bias = math.log(count1 / count0)
        output_bias = tf.keras.initializers.Constant(output_bias)

    kernel_init = model_config['WEIGHT_INITIALIZER']

    # Download 2D ResNet50
    if flow:
        base = ResNet50V2(include_top=False, input_tensor=None, input_shape=(input_shape[1:3] + [3]), pooling='avg')
    else:
        base = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape[1:],
                          pooling='avg')

    def block(x, last, filters, ind, pos, l2, kernel_init):

        '''
        Adds layers corresponding to a ResNet50 block (inflated to 3D)

        :param x: Output tensor from previous layers
        :param last: Last block's output tensor
        :param filters: Base number of filters in convolutions
        :param ind: Integer, tracking index of base ResNet50 layers (for naming purposes)
        :param pos: String, position of block in stage - either 'first', 'last', or a different arbitrary value
        :param l2: L2 regularization penalty, applied to all convolutional layers
        :param kernel_init: Kernel initialization method

        :return: Tensor - block output
        '''

        x = BatchNormalization(name=base.layers[ind].name)(x)
        x = Activation(activation='relu', name=base.layers[ind + 1].name)(x)

        if pos == 'first':
            last = x

        x = Conv3D(filters=filters, kernel_size=1, strides=1, kernel_initializer=kernel_init,
                   use_bias=False, kernel_regularizer=L2(l2), name=base.layers[ind + 2].name)(x)
        x = BatchNormalization(name=base.layers[ind + 3].name)(x)
        x = Activation(activation='relu', name=base.layers[ind + 4].name)(x)
        x = ZeroPadding3D(padding=1, name=base.layers[ind + 5].name)(x)

        if pos == 'last':
            x = Conv3D(filters=filters, kernel_size=3, strides=2, kernel_initializer=kernel_init,
                       use_bias=False, kernel_regularizer=L2(l2), name=base.layers[ind + 6].name)(x)
        else:
            x = Conv3D(filters=filters, kernel_size=3, strides=1, kernel_initializer=kernel_init,
                       use_bias=False, kernel_regularizer=L2(l2), name=base.layers[ind + 6].name)(x)
        x = BatchNormalization(name=base.layers[ind + 7].name)(x)
        x = Activation(activation='relu', name=base.layers[ind + 8].name)(x)

        # ADD STUFF
        if pos == 'first':
            res = Conv3D(filters=filters * 4, kernel_size=1, strides=1, kernel_initializer=kernel_init,
                         kernel_regularizer=L2(l2), name=base.layers[ind + 9].name)(last)
            x = Conv3D(filters=filters * 4, kernel_size=1, strides=1, kernel_initializer=kernel_init,
                       kernel_regularizer=L2(l2), name=base.layers[ind + 10].name)(x)
            x = Add(name=base.layers[ind + 11].name)([x, res])
        elif pos == 'last':
            res = MaxPooling3D(pool_size=1, strides=2, name=base.layers[ind + 9].name)(last)
            x = Conv3D(filters=filters * 4, kernel_size=1, strides=1, kernel_initializer=kernel_init,
                       kernel_regularizer=L2(l2), name=base.layers[ind + 10].name)(x)
            x = Add(name=base.layers[ind + 11].name)([x, res])
        else:
            x = Conv3D(filters=filters * 4, kernel_size=1, strides=1, kernel_initializer=kernel_init,
                       kernel_regularizer=L2(l2), name=base.layers[ind + 9].name)(x)
            x = Add(name=base.layers[ind + 10].name)([x, last])

        return x

    def stage(x, last, num_blocks, filters, ind, final, l2, kernel_init):

        '''
        Adds layers corresponding to a ResNet50 stage (inflated to 3D)

        :param x: Output tensor from previous layers
        :param last: Last block's output tensor
        :param num_blocks: Number of convolutional blocks in the stage
        :param filters: Base number of filters in convolutions
        :param ind: Integer, tracking index of base ResNet50 layers (for naming purposes)
        :param final: Boolean, flagging whether final stage or not (final stage has no dimensional reduction)
        :param l2: L2 regularization penalty
        :param kernel_init: Kernel initialization method

        :return: Tuple of (stage output tensor, index for next layer to be added)
        '''

        for i in range(num_blocks):
            pos = ''
            if i == 0:
                pos = 'first'
            elif i == (num_blocks - 1):
                if not final:
                    pos = 'last'
            x = block(x, last, filters, ind, pos, l2, kernel_init)
            last = x
            if (pos == 'first') or (pos == 'last'):
                ind += 12
            else:
                ind += 11
        return x, ind

    # USEFUL NOTE: Any conv with a BN directly after has no biases (handled by BN)

    # Input block
    inputs = tf.keras.Input(shape=input_shape)

    x = ZeroPadding3D(padding=3, name=base.layers[1].name)(inputs)
    x = Conv3D(filters=64, kernel_size=7, strides=2, kernel_initializer=kernel_init, kernel_regularizer=L2(l2_reg),
               name=base.layers[2].name)(x)
    x = ZeroPadding3D(padding=1, name=base.layers[3].name)(x)
    x = MaxPooling3D(pool_size=3, strides=2, name=base.layers[4].name)(x)

    index = 5

    x, index = stage(x, x, 3, 64, index, False, l2_reg, kernel_init)  # stage 1
    x, index = stage(x, x, 4, 128, index, False, l2_reg, kernel_init)  # stage 2
    x, index = stage(x, x, 6, 256, index, False, l2_reg, kernel_init)  # stage 3
    # x, index = stage(x, x, 3, 512, index, True, l2_reg, kernel_init)  # stage 4

    # Output head
    x = BatchNormalization(name=base.layers[-3].name)(x)
    x = Activation(activation='relu', name=base.layers[-2].name)(x)
    x = GlobalAveragePooling3D(name=base.layers[-1].name)(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation='sigmoid', kernel_initializer=kernel_init, bias_initializer=output_bias,
                    dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # Bootstrap weights & biases and freeze layers, only for convolutional and batch norm layers
    transfer = model_config['TRANSFER']
    if (not flow) and transfer:

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


def i3d(model_config, input_shapes, metrics, class_counts):
    '''
    Creates two-stream inflated inception, without transfer learning

    :param model_config: Hyperparameter dictionary
    :param input_shapes: Tuple of tuples, shapes of individual input tensors (without batch dimension)
        First shape corresponds to clip input shape
        Second shape corresponds to optical flow clip input shape
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    clip_shape = input_shapes[0]
    flow_shape = input_shapes[1]

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)

    dropout = model_config['DROPOUT']
    l2_reg = model_config['L2_REG']

    output_bias = None
    if cfg['TRAIN']['OUTPUT_BIAS']:
        count0 = class_counts[0]
        count1 = class_counts[1]
        output_bias = math.log(count1 / count0)
        output_bias = tf.keras.initializers.Constant(output_bias)

    kernel_init = model_config['WEIGHT_INITIALIZER']

    def inception_block(x, filters, l2):
        a = Conv3D(filters, kernel_size=(1, 1, 1), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), padding='same', use_bias=False)(x)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)

        b = Conv3D(filters, kernel_size=(1, 1, 1), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), padding='same', use_bias=False)(x)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Conv3D(filters, kernel_size=(3, 3, 3), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), padding='same', use_bias=False)(b)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)

        c = Conv3D(filters, kernel_size=(1, 1, 1), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), padding='same', use_bias=False)(x)
        c = BatchNormalization()(c)
        c = Activation('relu')(c)
        c = Conv3D(filters, kernel_size=(3, 3, 3), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), padding='same', use_bias=False)(c)
        c = BatchNormalization()(c)
        c = Activation('relu')(c)

        d = MaxPooling3D(pool_size=(3, 3, 3), strides=1, padding='same')(x)
        d = Conv3D(filters, kernel_size=(1, 1, 1), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), padding='same', use_bias=False)(d)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        return Concatenate()([a, b, c, d])

    def inception(x, l2):

        # number of filters from inception paper - very deep CNNs for large scale image rec

        x = Conv3D(64, kernel_size=(7, 7, 7), strides=2, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(x)

        x = Conv3D(128, kernel_size=(1, 1, 1), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(128, kernel_size=(3, 3, 3), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(x)

        for i in range(2):
            x = inception_block(x, 256, l2)

        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(x)
        '''
        for i in range(5):
            x = inception_block(x, 512, l2)

        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

        for i in range(2):
            x = inception_block(x, 512, l2)
        '''
        x = Conv3D(512, kernel_size=(1, 1, 1), strides=1, kernel_initializer=kernel_init,
                   kernel_regularizer=L2(l2), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = GlobalAveragePooling3D()(x)

        return x

    shape1 = clip_shape
    inputs1 = Input(shape1)
    x1 = inputs1
    x1 = inception(x1, l2_reg)
    m1 = tf.keras.Model(inputs=inputs1, outputs=x1)

    shape2 = flow_shape
    inputs2 = Input(shape2)
    x2 = inputs2
    x2 = inception(x2, l2_reg)
    m2 = tf.keras.Model(inputs=inputs2, outputs=x2)

    x = Concatenate()([m1.output, m2.output])
    x = Dropout(dropout)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid', kernel_initializer=kernel_init, bias_initializer=output_bias)(x)

    model = tf.keras.Model(inputs=[m1.input, m2.input], outputs=outputs)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model


def efficientnet(model_config, input_shape, metrics, class_counts):
    '''
    Defines a model based on pretrained EfficientNet. Assumes an M-mode reconstruction.

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)

    dropout = model_config['DROPOUT']
    l2_reg = model_config['L2_REG']
    fc0_nodes = model_config['FC0_NODES']

    freeze_cutoff = -1 if model_config['BLOCKS_FROZEN'] == 0 else model_config['BLOCKS_FROZEN']

    X_input = Input(input_shape, name='input')
    base_model = EfficientNet(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    output_bias = None
    kernel_init = model_config['WEIGHT_INITIALIZER']
    if cfg['TRAIN']['OUTPUT_BIAS']:
        count0 = class_counts[0]
        count1 = class_counts[1]
        output_bias = math.log(count1 / count0)
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Freeze layers
    for i in range(len(base_model.layers)):
        if i <= freeze_cutoff:
            base_model.layers[i].trainable = False
        # Freeze all batch norm layers
        elif ('batch' in base_model.layers[i].name) or ('bn' in base_model.layers[i].name):
            base_model.layers[i].trainable = False

    block_cutoff = ['top_activation', 'block6d_add', 'block5c_add', 'block4c_add']
    cutoff = block_cutoff[model_config['BLOCK_CUTOFF']]
    x = base_model.get_layer(cutoff).output

    # Add custom top layers
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    x = Dense(fc0_nodes, activation='relu', kernel_regularizer=L2(l2_reg), name='fc0')(x)

    x = Dense(1, bias_regularizer=output_bias, kernel_initializer=kernel_init, name='logits')(x)
    outputs = Activation('sigmoid', dtype='float32', name='output')(x)

    model = tf.keras.Model(inputs=X_input, outputs=outputs)
    model.summary()
    model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO,
                                                           alpha=model_config['ALPHA'], gamma=model_config['GAMMA']),
                  optimizer=optimizer, metrics=metrics)
    return model


def xception(model_config, input_shape, metrics, class_counts):
    '''
    Creates 2-dimensional Xception as specified by parameters in config file. Assumes an M-mode reconstruction.

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)

    dropout = model_config['DROPOUT']
    l2_reg = model_config['L2_REG']

    output_bias = None
    if cfg['TRAIN']['OUTPUT_BIAS']:
        count0 = class_counts[0]
        count1 = class_counts[1]
        output_bias = math.log(count1 / count0)
        output_bias = tf.keras.initializers.Constant(output_bias)

    kernel_init = model_config['WEIGHT_INITIALIZER']

    block_cutoffs = [6, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, -1]

    cutoff = block_cutoffs[model_config['BLOCKS'] - 1]

    if model_config['BLOCKS_FROZEN'] == 0:
        freeze_cutoff = -1
    else:
        freeze_cutoff = block_cutoffs[model_config['BLOCKS_FROZEN'] - 1]

    transfer = model_config['TRANSFER']

    if transfer:
        base_model = tf.keras.applications.Xception(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        base_model = tf.keras.applications.Xception(input_shape=input_shape, include_top=False, weights=None)

    # ----- Set regularization -------
    # Setting l2 regularization only sets it in the model config, and not in the layers
    # But reloading with the model config removes pre-trained weights
    # We have to save pre-trained weights, load by config (to get L2), and then load in weights
    # As per https://sthalles.github.io/keras-regularizer/

    for layer in base_model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, L2(l2_reg))

    # This json will contain added L2 regularization
    json = base_model.to_json()

    # Save weights
    weights_path = os.path.join(tempfile.gettempdir(), 'temp_weights.h5')
    base_model.save_weights(weights_path)

    # Load L2 regularization into layers (pre-trained weights will be lost here)
    base_model = tf.keras.models.model_from_json(json)

    # Load weights
    base_model.load_weights(weights_path, by_name=True)

    # --------------------------------

    # Take first n blocks as specified
    x = base_model.layers[cutoff].output

    # FC and output head
    x = GlobalAveragePooling2D()(x)
    x = Dense(16, activation='relu', kernel_regularizer=L2(l2_reg), bias_regularizer=L2(l2_reg))(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation='sigmoid', kernel_initializer=kernel_init, bias_initializer=output_bias,
                    dtype='float32')(x)

    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs)

    model.summary()

    # Freeze layers as specified if transfer learning is employed
    if transfer:
        for i in range(len(model.layers)):
            if i <= freeze_cutoff:
                model.layers[i].trainable = False
            # Freeze all batch norm layers
            elif ('batch' in model.layers[i].name) or ('bn' in model.layers[i].name):
                model.layers[i].trainable = False

    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO,
                                                           alpha=model_config['ALPHA'], gamma=model_config['GAMMA']),
                  optimizer=optimizer, metrics=metrics)

    return model


def vit(model_config, input_shape, metrics, class_counts):
    '''
    Creates 2-dimensional from-scratch vision transformer as specified by parameters in config file
    Reference: https://keras.io/examples/vision/image_classification_with_vision_transformer/

    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set

    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)

    dropout = model_config['DROPOUT']
    l2_reg = model_config['L2_REG']

    output_bias = None
    if cfg['TRAIN']['OUTPUT_BIAS']:
        count0 = class_counts[0]
        count1 = class_counts[1]
        output_bias = math.log(count1 / count0)
        output_bias = tf.keras.initializers.Constant(output_bias)

    kernel_init = model_config['WEIGHT_INITIALIZER']

    # Calculate resize and patches
    orig_size = [input_shape[0], input_shape[1]]
    resize_height = None
    resize_width = None
    if orig_size[0] > orig_size[1]:
        resize_height = int(math.ceil(orig_size[0] / orig_size[1]) * orig_size[1])
    else:
        resize_width = int(math.ceil(orig_size[1] / orig_size[0]) * orig_size[0])
    resize_shape = [resize_height, orig_size[1]] if resize_height else [orig_size[0], resize_width]
    patch_size = 10  # Size of the patches to be extract from the input images
    num_patches = (resize_shape[0] // patch_size) * (resize_shape[1] // patch_size)

    # Other ViT-related params
    projection_dim = model_config['PROJECTION_DIM']
    num_heads = model_config['NUM_HEADS']
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = model_config['TRANSFORMER_LAYERS']
    layer_norm_epsilon = model_config['LAYER_NORM_EPSILON']

    class Patches(tf.keras.layers.Layer):
        '''
        Creates image patches from input image batch
        '''

        def __init__(self, patch_size):
            super(Patches, self).__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

        def compute_output_shape(input_shape):
            return [1, 1, 1, 1]

    def mlp(x, hidden_units, dropout_rate, l2_reg):
        '''
        Applies a specified number of MLP units (where each unit is an FC layer with a GELU activation) to a tensor

        :param x: Input tensor
        :param hidden_units: Number of MLP units to apply
        :param dropout_rate: Dropout rate from GELU to FC
        :param l2_reg: L2 regularization penalty

        :return: Output tensor from final MLP unit
        '''

        for units in hidden_units:
            x = Dense(units, activation=tf.nn.gelu, kernel_regularizer=L2(l2_reg), bias_regularizer=L2(l2_reg))(x)
            x = Dropout(dropout_rate)(x)
        return x

    class PatchEncoder(tf.keras.layers.Layer):
        '''
        Embeds image patches and adds positional embeddings
        '''

        def __init__(self, num_patches, projection_dim):
            super(PatchEncoder, self).__init__()
            self.num_patches = num_patches
            self.projection = Dense(units=projection_dim)
            self.position_embedding = Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

    # Resize images and get patch embeddings
    inputs = tf.keras.Input(shape=input_shape)
    resized = Resizing(resize_shape[0], resize_shape[1])(inputs)
    patches = Patches(patch_size)(resized)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Apply transformer architecture to embeddings
    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=layer_norm_epsilon)(encoded_patches)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout)(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=layer_norm_epsilon)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout, l2_reg=l2_reg)
        encoded_patches = Add()([x3, x2])

    # Output head
    x = LayerNormalization(epsilon=layer_norm_epsilon)(encoded_patches)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation='sigmoid', kernel_initializer=kernel_init, bias_initializer=output_bias,
                    dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model


def variance_mmode_net(model_config, input_shape, metrics, class_counts):
    '''
    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set
    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)

    model = Sequential([Input((224, 90, 3))])
    model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
    model.add(Reshape((224, 90)))
    model.add(Lambda(lambda x: tf.math.reduce_variance(x, axis=2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model


def one_d_conv(model_config, input_shape, metrics, class_counts):
    '''
    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set
    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)

    model = Sequential([Input((224, 90, 3))])
    model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
    model.summary()
    model.add(Conv1D(64, kernel_size=3, activation='relu', strides=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu', strides=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu', strides=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu', strides=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu', strides=2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid', dtype='float32'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model


def mobile_net_v3_small(model_config, input_shape, metrics, class_counts):
    '''
    :param model_config: Hyperparameter dictionary
    :param input_shape: Tuple, shape of individual input tensor (without batch dimension)
    :param metrics: List of metrics for model compilation
    :param class_counts: 2-element list - number of each class in training set
    :return: Compiled tensorflow model
    '''

    lr = model_config['LR']
    optimizer = Adam(learning_rate=lr)
    model = Sequential([])
    base_model = MobileNetV3Small(input_shape=input_shape,
                                  include_top=False,
                                  weights='imagenet',
                                  include_preprocessing=False)

    model = Sequential([base_model, GlobalAveragePooling2D(), Dense(1, activation='sigmoid')])
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model
