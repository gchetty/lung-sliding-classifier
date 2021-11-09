
from re import S
from tensorflow.python.keras.metrics import TrueNegatives, TruePositives
import yaml
import os
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.metrics import Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives, Accuracy
from tensorflow_addons.metrics import F1Score, FBetaScore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocessor import Preprocessor, FlowPreprocessor
from visualization.visualization import log_confusion_matrix
from models.models import *
from custom.metrics import Specificity
from data.utils import refresh_folder

model_path = 'results/models/res3d/'
custom_objects = {'Specificity': Specificity, 'FBetaScore': FBetaScore}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

sliding1 = np.load('sample_1.npz', allow_pickle=True)['frames']
sliding1 = sliding1[None,:,:,:,:].astype(np.float32)
sliding2 = np.load('sample_2.npz', allow_pickle=True)['frames']
sliding2 = sliding2[None,:,:,:,:].astype(np.float32)

clips = [sliding1, sliding2]
labels = np.array([1, 1]).astype(np.uint8)
dataset = tf.data.Dataset.from_tensor_slices((clips, labels))
dataset.map(lambda x, y: ((x / 255.0), y))
dataset.batch(2)

last_conv = 'conv3d_7'
inp = sliding1

with tf.GradientTape() as tape:
  last_conv_layer = model.get_layer(last_conv)
  print(last_conv_layer.output_shape)
  tape.watch(last_conv_layer.variables)
  #print(last_conv_layer.get_weights()[0])  # there are non-zero weights loaded
  iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
  model_out, last_conv_layer = iterate(inp)
  class_out = model_out
  #print(class_out)
  #print(last_conv_layer)
  grads = tape.gradient(class_out, last_conv_layer)
  print(grads.shape)
  print(tf.math.count_nonzero(grads))  # all gradients are zero ???
  pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2, 3))
  print(pooled_grads.shape)
  print(pooled_grads)
