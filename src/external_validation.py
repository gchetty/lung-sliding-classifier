import gc
import glob
import math
import os
import argparse
import random

import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.losses import SigmoidFocalCrossEntropy, sigmoid_focal_crossentropy

from models.models import get_model
from src.custom.metrics import Specificity
from src.data.sql_utils import add_date_to_filename
from src.data.utils import refresh_folder
from src.preprocessor import MModePreprocessor
from train import log_train_params

# cfg is the sub-dictionary of the config file pertaining to fine-tuning experiments. cfg_full is the full config file.
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))

SLIDING = 1
NO_SLIDING = 0


def make_scheduler(writer, hparams):
    '''
    Creates a scheduler function that takes in two arguments: epoch and learning rate.
    :param writer:
    :param hparams: Set of hyperparameters to use in the created scheduler.
    :returns: Function. A new scheduler is created.
    '''
    def scheduler(epoch, lr):
        '''
        Returns learning rate for the upcoming epoch based on a set schedule
        Decreases learning rate by a factor of e^-(DECAY_VAL) starting at epoch 15

        :param epoch: Integer, training epoch number
        :param lr: Float, current learning rate

        :return: Float, new learning rate
        '''
        learning_rate = lr
        if epoch > hparams['LR_DECAY_EPOCH']:
            learning_rate = lr * tf.math.exp(-1 * hparams['LR_DECAY_VAL'])
        with writer.as_default():  # Write LR scalar to log directory
            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate
    return scheduler


def write_folds_to_csv(folds, file_path):
    '''
    Saves clip id folds to .csv file in the following structure:
    -> Contains all columns from the db for the given clip ids in specified fold.
    -> Columns preceded by an additional column, which specifies fold number.
    :param folds: List of pd.DataFrames. Each DataFrame is a fold.
    :param file_path: Absolute path where folds should be stored.
    '''

    folds_df = pd.DataFrame([])
    for i in range(len(folds)):
        folds[i].insert(0, 'fold_number', i + 1)
        folds_df = pd.concat([folds_df, folds[i]], ignore_index=True)
    folds_df.to_csv(file_path, index=False)


def check_performance(df):
    '''
    Returns true if all metrics stored in the supplied results DataFrame meet fine-tuning standards, or False otherwise.
    :param df: DataFrame storing model performance metrics to be evaluated against fine-tune performance thresholds.
    :returns: Bool, whether metrics in df satisfy performance threshold conditions.
    '''
    cfg_thresh = cfg['FINETUNE']['METRIC_THRESHOLDS']

    # Check that metrics exceed lower bounds, if they exist
    for thresh in cfg_thresh['LOWER_BOUNDS']:
        if df[thresh.lower()].values[0] < cfg_thresh['LOWER_BOUNDS'][thresh]:
            return False

    # Check that metrics do not exceed upper bounds, if they exist
    for thresh in cfg_thresh['UPPER_BOUNDS']:
        if df[thresh.lower()].values[0] > cfg_thresh['UPPER_BOUNDS'][thresh]:
            return False

    # If all metric conditions are satisfied, return True.
    return True


def get_external_clip_df():
    '''
    Returns combined df containing results from raw external db pulls for all multi-center study locations.
    :returns: pd.DataFrame. Stores combined clip information as shown in external db.
    '''
    ext_dfs = []
    for center in cfg['EXTERNAL_VAL']['LOCATIONS']:
        center_split_path = os.path.join(os.getcwd() + cfg['EXTERNAL_VAL']['PATHS']['CSVS_SPLIT'], center)
        for csv in ['train.csv', 'val.csv']:
            ext_dfs.append(pd.read_csv(os.path.join(center_split_path, csv)))
    ext_df = pd.concat(ext_dfs, ignore_index=True)
    labelled_ext_df_path = os.path.join(os.getcwd() + cfg['EXTERNAL_VAL']['PATHS']['CSV_OUT'], 'labelled_external_dfs/')
    if cfg['EXTERNAL_VAL']['REFRESH_FOLDERS']:
        refresh_folder(labelled_ext_df_path)
    labelled_ext_df_path += add_date_to_filename('labelled_ext') + '.csv'
    ext_df.to_csv(labelled_ext_df_path)

    return ext_df


def evaluate_model(model, test_df, preprocessor, base_folder):
    '''
    Returns a df containing evaluation results of the model stored at model_path on test_set, using specified metrics.
    :param model: Model to evaluate; precompiled with proper metrics
    :param test_df: TF dataset on which to evaluate the model.
    :param preprocessor: An MMode preprocessor object
    :param base_folder: Folder in which to log predictions
    :return: dict of evaluation metrics
    '''

    test_set = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label'].tolist()))
    preprocessed_test_set = preprocessor.prepare(test_set, test_df, shuffle=False, augment=False)

    # Determine results for metrics
    test_results = model.evaluate(preprocessed_test_set, return_dict=True)

    test_pred_probs = model.predict(preprocessed_test_set)
    test_pred_class = np.round(test_pred_probs)
    test_results['confusion_matrix'] = str(confusion_matrix(y_true=test_df['label'], y_pred=test_pred_class).ravel())

    # Save predictions
    test_preds_df = test_df[['filename', 'label']].copy(deep=True)
    test_preds_df['pred_prob'] = test_pred_probs
    test_preds_df['pred_class'] = test_pred_class
    test_preds_df.to_csv(os.path.join(base_folder, 'test_set_preds.csv'), index=False)

    return test_results


def setup_experiment(experiment_path=None):
    '''
    Creates a new experiment folder if no path is supplied, identified with a Unix timestamp. Then creates a
    subdirectory for a new trial in the experiment. Returns the path to the newly created trial.
    :param experiment_path: Absolute path to folder storing experiments.
    :returns: Absolute path to newly created experiment.
    '''

    if experiment_path is None:
        experiment_folder = os.getcwd() + cfg['EXTERNAL_VAL']['PATHS']['EXPERIMENTS']
        experiment_path = os.path.join(experiment_folder, add_date_to_filename('experiment'))
        os.makedirs(experiment_path)
    else:
        assert os.path.exists(experiment_path), f"{experiment_path} does not exist."

    # Create a new trial folder
    n_existing_trials = len(os.listdir(experiment_path))
    new_trial_num = n_existing_trials + 1
    trial_path = os.path.join(experiment_path, f'trial_{new_trial_num}')
    os.makedirs(trial_path)
    return trial_path


def sample_folds(clip_df, trial_folder, k, seed=None):
    '''
    Samples external data in clip_df into k folds. Attempts to preserve ratio of positive to negative classes in each
    fold. Folds are saved in a .csv file, where each fold is identified with an integer index, followed by each clip
    id in the fold
    :param clip_df: Clip table for all clips in this trial
    :param trial_folder: Absolute path to folder in which to place folds. Corresponds to a particular trial.
    :param k: Number of folds
    :param seed: Random seed for fold sampling
    '''

    sgkfold = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for _, fold_idxs in sgkfold.split(clip_df, clip_df['label'], clip_df['patient_id'].astype(str)):
        next_fold = clip_df.iloc[fold_idxs]
        folds.append(next_fold)
    random.shuffle(folds)

    # Print fold characteristics
    for i in range(len(folds)):
        counts = folds[i]['label'].value_counts()
        print(f"Fold {i}: {counts[1]} sliding. {counts[0]} absent sliding. absent/total = {counts[0] / folds[i].shape[0]}")

    # Save folds for current trial
    folds_path = os.path.join(trial_folder, 'folds.csv')
    write_folds_to_csv(folds, folds_path)
    return folds


def restore_model(original_model_path):
    '''
    Loads original model weights from disk and compiles it with metrics of interest.
    :param original_model_path: Path to folder containing the original model weights (in TF SavedModel format)
    :return: tf.keras model with original weights
    '''

    original_model = load_model(original_model_path, compile=False)

    metrics = [Recall(name='sensitivity'), Specificity(name='specificity')]
    model_config = cfg['TRAIN']['PARAMS']['EFFICIENTNET']
    lr = cfg['EXTERNAL_VAL']['FINETUNE']['LR']
    optimizer = Adam(learning_rate=lr)

    # Compile the model using specified list of metrics.
    original_model.compile(loss=SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO,
                                                alpha=model_config['ALPHA'],
                                                gamma=model_config['GAMMA']),
                  optimizer=optimizer, metrics=metrics)
    return original_model


def freeze_model_layers(model, last_frozen_layer):
    '''
    Freeze all layers in the model up until a specified layer index. Ensure that all batch norm layers are
    in inference mode (i.e. training=False).
    :param model: A tf.keras model object
    :param last_frozen_layer: The index of the last layer to freeze in the model
    :return: A tf.keras model object with some layers frozen
    '''
    for i in range(len(model.layers)):
        if i <= last_frozen_layer:
            model.layers[i].trainable = False
        elif ('batch' in model.layers[i].name) or ('bn' in model.layers[i].name):
            model.layers[i].trainable = False           # Ensure all batch norm layers are frozen
    return model


def finetune_model(original_model_path, base_folder, full_train_df, mmode_preprocessor, hparams, upsample=True, seed=None):
    '''
    Fine-tunes the model on a training set.
    :param original_model_path: Path to SavedModel constituting the original weights
    :param base_folder: The folder in which logs, models, and train/val splits will be saved
    :param full_train_df: The training set (before splitting off a validation set)
    :param mmode_preprocessor: a MModePreprocessor object for preparing batches
    :param hparams: Model hyperparameters
    :param upsample: If set to True, upsamples the minority class in the training set according to the configured ratio
    :param seed: Random seed for fold sampling
    '''

    # Split dataset into training and validation sets
    val_split = cfg['EXTERNAL_VAL']['FINETUNE']['VAL_SPLIT']
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = splitter.split(full_train_df, full_train_df['label'], full_train_df['patient_id'].astype(str)).__next__()
    train_df = full_train_df.iloc[train_idx]
    val_df = full_train_df.iloc[val_idx]

    # Oversample the minority class in the training set
    minority_frac = cfg['EXTERNAL_VAL']['FINETUNE']['MINORITY_FRAC']
    if upsample and (train_df['label'].value_counts().min() / train_df.shape[0] < minority_frac):
        # minority_ratio = 1. / (1. - minority_frac) - 1
        # oversampler = RandomOverSampler(sampling_strategy=minority_ratio, random_state=seed)
        # train_df, _ = oversampler.fit_resample(train_df, train_df['label'])
        counts = train_df['label'].value_counts()
        n_additional = math.ceil((1. / (1. - minority_frac) - 1) * counts.max()) - counts.min()
        absent_train_df = train_df.loc[train_df['label'] == 0].copy().sample(frac=1, random_state=seed)
        extra_idx = 1
        for i in range(n_additional):
            if i % absent_train_df.shape[0] == 0:
                extra_idx += 1
            extra_absent_example = absent_train_df.iloc[i % absent_train_df.shape[0]]
            extra_absent_example['filename'] = os.path.splitext(extra_absent_example['filename'])[0] + f'_{extra_idx}.npz'
            train_df = train_df.append(extra_absent_example)

    # Save training and validation sets
    splits_dir = os.path.join(base_folder, 'splits')
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    print(f"TRAIN SET: {train_df['label'].value_counts().to_dict()}, VAL_SET: {val_df['label'].value_counts().to_dict()}")

    # Prepare TF datasets
    train_set = tf.data.Dataset.from_tensor_slices((train_df['filename'].tolist(), train_df['label'].tolist()))
    train_set = mmode_preprocessor.prepare(train_set, train_df, shuffle=True, augment=True)
    val_set = tf.data.Dataset.from_tensor_slices((val_df['filename'].tolist(), val_df['label'].tolist()))
    val_set = mmode_preprocessor.prepare(val_set, val_df, shuffle=False, augment=False)

    # Get class weights for loss weighting
    counts = train_df['label'].value_counts()
    weights = train_df.shape[0] / (2.0 * np.array([counts[0], counts[1]]))
    class_weight = {0: weights[0], 1: weights[1]}

    # Set up TensorBoard logs
    tensorboard_path = os.path.join(base_folder, 'logs')
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
    writer = tf.summary.create_file_writer(tensorboard_path + '/train')
    if tensorboard_path is not None:
        log_train_params(writer, hparams)

    # Create learning rate scheduler
    lr_scheduler = make_scheduler(writer, hparams)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min',
                                   restore_best_weights=True, min_delta=cfg['EXTERNAL_VAL']['FINETUNE']['MIN_DELTA'])

    # Restore weights of original model
    model = restore_model(original_model_path)

    # Freeze model layers
    model = freeze_model_layers(model, 220)

    # Initialize optimizer, metrics, and compile model
    optimizer = Adam(learning_rate=cfg['EXTERNAL_VAL']['FINETUNE']['LR'] / 10.)
    metrics = [Recall(name='sensitivity'), Specificity(name='specificity')]
    model.compile(loss=SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO,
                                                alpha=hparams['ALPHA'], gamma=hparams['GAMMA']),
                  optimizer=optimizer, metrics=metrics)

    # Train top layer
    model.fit(train_set, epochs=10, validation_data=val_set,
                class_weight=class_weight,
                callbacks=[tensorboard, lr_callback], verbose=1)

    # print("Unfreezing part of base model")
    # freeze_cutoff = -1 if hparams['LAYERS_FROZEN'] == 0 else hparams['LAYERS_FROZEN']
    # if freeze_cutoff > 0:
    #     model = freeze_model_layers(model, freeze_cutoff)
    # optimizer = Adam(learning_rate=cfg['EXTERNAL_VAL']['FINETUNE']['LR'] / 10.)
    # model.compile(loss=SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO,
    #                                             alpha=hparams['ALPHA'], gamma=hparams['GAMMA']),
    #               optimizer=optimizer, metrics=metrics)
    # model.fit(train_set, epochs=cfg['EXTERNAL_VAL']['FINETUNE']['EPOCHS'], validation_data=val_set,
    #             class_weight=class_weight, initial_epoch=10,
    #             callbacks=[tensorboard, early_stopping, lr_callback], verbose=1)

    # Save and return best model's weights
    model_path = os.path.join(base_folder, f'model')
    model.save(model_path)
    return model


def TAAFT(clip_df, n_folds, experiment_path=None, seed=None, hparams=None, lazy=True, upsample=True):
    '''
    Executes a trial of TAAFT (Threshold-Aware Accumulative Fine-Tuning).
    Divides clips into random folds, then runs fine-tuning trials with progressively larger training sets. Evaluates
    each fine-tuned model on remaining data (variable-size test set and fixed-size test set).
    :param clip_df: A DataFrame consisting of all clips in the dataset on which to fine-tune the model
    :param experiment_path: Absolute path to folder corresponding to the current experiment
    :param seed: Random seed for experiment replicability
    :param hparams: Specifies the set of hyperparameters for fine-tuning the model.
    :param lazy: When True, trial will terminate once model performance exceeds minimum thresholds.
    :param upsample: When True, upsamples absent sliding mini-clips in each training set during fine-tuning.
    '''

    # Set random seeds
    if seed is not None:
        random.seed(seed)
        tf.random.set_seed(seed)

    trial_folder = setup_experiment(experiment_path)                    # Set up folder for this trial
    folds = sample_folds(clip_df, trial_folder, n_folds, seed=seed)     # Randomly sample folds

    original_model_path = cfg['PREDICT']['MODEL']

    # This df will store trial results.
    trial_results_df = pd.DataFrame([])

    # Initialize preprocessor for M-Mode images
    _, preprocessing_fn = get_model(cfg['EXTERNAL_VAL']['FINETUNE']['MODEL_DEF'])
    mmode_preprocessor = MModePreprocessor(preprocessing_fn)

    # Define the fixed test set
    n_folds_fixed_test_set = math.ceil(cfg['EXTERNAL_VAL']['FIXED_TEST_SIZE'] *
                                       cfg['EXTERNAL_VAL']['FOLD_SAMPLE']['NUM_FOLDS'])
    fixed_test_df = pd.concat(folds[-n_folds_fixed_test_set:], axis=0)

    # Get model hyperparameters
    hparams = cfg['TRAIN']['PARAMS'][cfg['TRAIN']['MODEL_DEF'].upper()] if hparams is None else hparams
    hparams['seed'] = seed

    # Conduct fine tuning experiments for progressively larger training sets
    for i in range(0, n_folds - n_folds_fixed_test_set + 1):

        train_frac = i / n_folds
        base_folder = os.path.join(trial_folder, str(train_frac))
        os.makedirs(base_folder)

        # Initialize train, variable test, and fixed test sets
        variable_test_df = pd.concat(folds[i:n_folds], axis=0)

        # Restore the model's weights and fine-tune the model
        if i == 0:
            model = restore_model(original_model_path)
        else:
            print(f"**** TRAINING RUN {i}: train_frac={train_frac} ****\n" + 100 * '*')
            train_df = pd.concat(folds[:i])
            model = finetune_model(original_model_path, base_folder, train_df, mmode_preprocessor, hparams,
                                   upsample=upsample, seed=seed)

        # Evaluate model on fixed test set
        results_row = {'variable_size_test_set': 1, 'train_data_prop': i / n_folds}
        results_dict = evaluate_model(model, variable_test_df, mmode_preprocessor, base_folder)
        results_row.update(results_dict)
        trial_results_df = trial_results_df.append(results_row, ignore_index=True)

        # Evaluate model on variable test set
        results_row = {'variable_size_test_set': 0, 'train_data_prop': i / n_folds}
        results_dict = evaluate_model(model, fixed_test_df, mmode_preprocessor, base_folder)
        results_row.update(results_dict)
        trial_results_df = trial_results_df.append(results_row, ignore_index=True)

        # Relase unused memory and reset the TF session
        gc.collect()
        tf.keras.backend.clear_session()
        del model

    trial_results_df.to_csv(os.path.join(trial_folder, 'metrics.csv'), index=False)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract mini-clips as npzs")
    parser.add_argument('--experiment', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int)
    args = parser.parse_args()

    # If the generalizability folder has not been created, create the subdirectories for each of the
    # required data types (e.g. raw clips, .npzs, m-modes, etc.)
    for path in cfg['EXTERNAL_VAL']['PATHS']:
        if not os.path.exists(os.getcwd() + cfg['EXTERNAL_VAL']['PATHS'][path]):
            os.makedirs(os.getcwd() + cfg['EXTERNAL_VAL']['PATHS'][path])

    # Assemble miniclips from all external centres
    external_df = get_external_clip_df()

    # Conduct fine-tuning experiment
    seed = args.seed if args.seed is not None else cfg['EXTERNAL_VAL']['FOLD_SAMPLE']['SEED']
    experiment_path = args.experiment
    TAAFT(external_df, cfg['EXTERNAL_VAL']['FOLD_SAMPLE']['NUM_FOLDS'], experiment_path, seed, upsample=True)
