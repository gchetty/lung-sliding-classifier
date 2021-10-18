'''
Script for training experiments, including model training, hyperparameter search, cross validation, etc
'''

import yaml
import os
import pandas as pd
import tensorflow as tf
from preprocessor import Preprocessor
from models.models import *


from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import ModelCheckpoint


cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def train_model(model_def, hparams, preprocessing_fn=(lambda x: x), save_weights=False, log_dir=None, verbose=True):

    model_def, preprocessing_fn = get_model()

    # Read in train, val, test dataframes
    CSVS_FOLDER = cfg['TRAIN']['PATHS']['CSVS']
    train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'train.csv'))
    val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'val.csv'))
    test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'test.csv'))

    # Create TF datasets for training, validation and test sets
    train_set = tf.data.Dataset.from_tensor_slices((train_df['filename'].tolist(), train_df['label']))
    val_set = tf.data.Dataset.from_tensor_slices((val_df['filename'].tolist(), val_df['label']))
    test_set = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label']))

    preprocessor = Preprocessor(preprocessing_fn)

    train_set = preprocessor.prepare(train_set, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, shuffle=False, augment=False)
    test_set = preprocessor.prepare(test_set, shuffle=False, augment=False)

    num_no_sliding = len(train_df[train_df['label']==0])
    num_sliding = len(train_df[train_df['label']==1])
    total = num_no_sliding + num_sliding

    # Taken from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / num_no_sliding) * (total / 2.0)
    weight_for_1 = (1 / num_sliding) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(class_weight)

    # Metrics
    classes = [0, 1]
    n_classes = len(classes)
    threshold = 1.0 / n_classes
    metrics = ['accuracy', AUC(name='auc'), F1Score(name='f1score', num_classes=2)]
    metrics += [Precision(name='precision_' + str(classes[i]), thresholds=threshold, class_id=i) for i in range(n_classes)]
    metrics += [Recall(name='recall_' + str(classes[i]), thresholds=threshold, class_id=i) for i in range(n_classes)]

    save_cp = ModelCheckpoint('models/model1', save_best_only=True)

    # Get model
    hparams = cfg['TRAIN']['PARAMS']['TEST1']  # TEMPORARY LOCATION
    input_shape = [cfg['PREPROCESS']['PARAMS']['WINDOW']] + cfg['PREPROCESS']['PARAMS']['IMG_SIZE'] + [3]
    model = model_def(hparams, input_shape, metrics, n_classes)  # do we need output bias?

    model.fit(train_set, epochs=15, validation_data=val_set, class_weight=class_weight, callbacks=[save_cp])

train_model(None, None)
