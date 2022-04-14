'''
Script for training experiments, including model training, hyperparameter search, cross validation, etc
'''
import keras.callbacks
import logging
import numpy as np
import tensorflow.keras.backend
import yaml
import os
import datetime
import pandas as pd
import tensorflow as tf

from tensorflow.keras.metrics import Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, \
    FalsePositives, Accuracy, SensitivityAtSpecificity
from tensorflow_addons.metrics import F1Score, FBetaScore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from preprocessor import Preprocessor, FlowPreprocessor, TwoStreamPreprocessor, MModePreprocessor
from visualization.visualization import plot_bayesian_hparam_opt, plot_roc, plot_to_tensor, plot_confusion_matrix
from models.models import *
from custom.metrics import Specificity, PhiCoefficient
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt import load
import gc

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def log_test_results(model, test_set, test_df, test_metrics, writer):
    '''
    Visualize performance of a trained model on the test set. Optionally save the model.
    :param model: A trained TensorFlow model
    :param test_set: A TensorFlow dataset for the test set
    :param test_df: Dataframe containing npz files and labels for test set
    :param test_metrics: Dict of test set performance metrics
    :param writer: file writer object for tensorboard
    '''

    # Visualization of test results
    test_predictions = model.predict(test_set, verbose=0)
    labels = test_df['label'].to_numpy()
    plt = plot_roc(labels, test_predictions, [0, 1])
    roc_img = plot_to_tensor()
    plt = plot_confusion_matrix(labels, test_predictions, [0, 1])
    cm_img = plot_to_tensor()

    # Create table of test set metrics
    test_summary_str = [['**Metric**', '**Value**']]
    for metric in test_metrics:
        metric_values = test_metrics[metric]
        test_summary_str.append([metric, str(metric_values)])

    # Write to TensorBoard logs
    with writer.as_default():
        tf.summary.text(name='Test set metrics', data=tf.convert_to_tensor(test_summary_str), step=0)
        tf.summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
        tf.summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)
    return


def log_miniclip_test_results(model, test_set, test_df, writer):
    '''
    Adjust and log clip and mini-clip prediction results - if any mini-clip is sliding, then entire clip is sliding
    :param model: A trained TensorFlow model
    :param test_set: A TensorFlow dataset for the test set
    :param test_df: Dataframe containing npz files and labels for test set
    :param writer: file writer object for tensorboard
    '''

    # Get predictions
    test_predictions = model.predict(test_set, verbose=0)
    df_copy = test_df.copy()

    # get parent clip ids
    clip_ids = []
    for idx, row in test_df.iterrows():
        clip_ids.append(row['id'].split('_')[0])
    df_copy.loc[:, 'clip_id'] = clip_ids
    df_copy.loc[:, 'pred'] = np.where(test_predictions >= 0.5, 1, 0)

    # go through each parent clip and reassign miniclip predictions (if any miniclip is sliding, all are sliding)
    # also create labels and predictions for clips only
    clips = pd.unique(df_copy['clip_id'])
    clip_labels = []
    clip_preds = []
    for clip_id in clips:
        # get all miniclips belonging to parent clip
        miniclips = df_copy[df_copy['clip_id'] == clip_id]

        # assing label of 1 to all miniclips if any miniclip has label of 1
        if any(miniclips['pred'] == 1):
            df_copy.loc[df_copy['clip_id'] == clip_id, 'pred'] = [1] * len(miniclips)

        clip_labels.append(miniclips.iloc[0]['label'])
        clip_preds.append(miniclips.iloc[0]['pred'])

    metrics = [Accuracy, AUC, Recall, Specificity, TrueNegatives, TruePositives, FalseNegatives, FalsePositives]

    # Create table of metrics for adjusted clips and miniclips
    miniclip_summary_str = [['**Metric**', '**Value**']]
    clip_summary_str = [['**Metric**', '**Value**']]
    for metric in metrics:
        m = metric()
        m.update_state(df_copy['label'], df_copy['pred'])
        miniclip_summary_str.append([m.name, str(m.result().numpy())])

        m.reset_state()
        m.update_state(clip_labels, clip_preds)
        clip_summary_str.append([m.name, str(m.result().numpy())])

    # Write to TensorBoard logs
    with writer.as_default():
        tf.summary.text(name='Test set metrics for clips', data=tf.convert_to_tensor(clip_summary_str), step=0)
        tf.summary.text(name='Test set metrics for adjusted mini-clips',
                        data=tf.convert_to_tensor(miniclip_summary_str), step=0)
    return


def log_train_params(writer, hparams):
    """
    Log hyperparameters and early stopping information to tensorboard
    :param writer: file writer object for tensorboard
    :param hparams: list of hyperparameters used for the model
    """

    # Create training parameters table
    train_summary_str = [['**Train Parameter**', '**Value**']]
    for key in cfg['TRAIN']:
        if key == 'PARAMS':
            for param in cfg['TRAIN']['PARAMS']:
                if param == 'AUGMENTATION':
                    break
                train_summary_str.append([param, str(cfg['TRAIN']['PARAMS'][param])])
            break
        train_summary_str.append([key, str(cfg['TRAIN'][key])])

    # Create augmentation parameters table
    aug_summary_str = [['**Augmentation Parameter**', '**Value**']]
    for key in cfg['TRAIN']['PARAMS']['AUGMENTATION']:
        aug_summary_str.append([key, str(cfg['TRAIN']['PARAMS']['AUGMENTATION'][key])])

    # Create hyperparameter table
    hparam_summary_str = [['**Hyperparameter**', '**Value**']]
    for key in hparams:
        hparam_summary_str.append([key, str(hparams[key])])

    # Write to TensorBoard logs
    with writer.as_default():
        tf.summary.text(name='Run Train Params', data=tf.convert_to_tensor(train_summary_str), step=0)
        tf.summary.text(name='Run Augmentation Params', data=tf.convert_to_tensor(aug_summary_str), step=0)
        tf.summary.text(name='Run hyperparameters', data=tf.convert_to_tensor(hparam_summary_str), step=0)
    return


def train_model(model_def_str=cfg['TRAIN']['MODEL_DEF'],
                hparams=cfg['TRAIN']['PARAMS']['XCEPTION'],
                model_out_dir=cfg['TRAIN']['PATHS']['MODEL_OUT'], train_df=None, val_df=None, log_name=None):
    '''
    Trains and saves a model given specific hyperparameters

    :param model_def_str: A string in {'test1', ... } specifying the name of the desired model
    :param hparams: A dictionary specifying the hyperparameters to use
    :param model_out_dir: The path to save the model
    '''

    # hparams = cfg['TRAIN']['PARAMS'][model_def_str.upper()]

    # Enable mixed precision
    mixed_precision = cfg['TRAIN']['MIXED_PRECISION']
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    flow = cfg['PREPROCESS']['PARAMS']['FLOW']
    m_mode = cfg['TRAIN']['M_MODE']

    # Get the model function and preprocessing function 
    model_def_fn, preprocessing_fn = get_model(model_def_str)

    CSVS_FOLDER = cfg['TRAIN']['PATHS']['CSVS']

    test_df = None
    test_set = None

    if train_df is None or val_df is None:

        # Read in training, validation, and test dataframes
        if flow == 'Yes':
            train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'flow_train.csv'))  # [:20]
            val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'flow_val.csv'))  # [:20]
            test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'flow_test.csv'))  # [:20]
        else:
            train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'train.csv'))  # [:20]
            val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'val.csv'))  # [:20]
            test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'test.csv'))  # [:20]

    # Create TF datasets for training, validation and test sets
    # Note: This does NOT load the dataset into memory! We specify paths,
    #       and labels so that TensorFlow can perform dynamic loading.
    train_set = tf.data.Dataset.from_tensor_slices((train_df['filename'].tolist(), train_df['label']))
    val_set = tf.data.Dataset.from_tensor_slices((val_df['filename'].tolist(), val_df['label']))
    if cfg['TRAIN']['SPLITS']['TEST'] > 0:
        test_set = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label']))

    # Create preprocessing object given the preprocessing function for model_def
    if m_mode:
        preprocessor = MModePreprocessor(preprocessing_fn)
    elif flow == 'Yes':
        preprocessor = FlowPreprocessor(preprocessing_fn)
    else:
        preprocessor = Preprocessor(preprocessing_fn)

    # Define the preprocessing pipelines for train, test and validation
    train_set = preprocessor.prepare(train_set, train_df, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, val_df, shuffle=False, augment=False)
    if cfg['TRAIN']['SPLITS']['TEST'] > 0:
        test_set = preprocessor.prepare(test_set, test_df, shuffle=False, augment=False)

    # Create dictionary for class weights:
    # Taken from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    num_no_sliding = len(train_df[train_df['label'] == 0])
    num_sliding = len(train_df[train_df['label'] == 1])
    total = num_no_sliding + num_sliding
    weight_for_0 = (1 / num_no_sliding) * (total / 2.0)
    weight_for_1 = (1 / num_sliding) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    counts = [num_no_sliding, num_sliding]

    # Defining Binary Classification Metrics
    metrics = ['accuracy', AUC(name='auc'), FBetaScore(num_classes=2, average='micro', threshold=0.5)]
    metrics += [Precision(name='precision'), Recall(name='recall')]
    metrics += [TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives(), Specificity(name='specificity'),
                PhiCoefficient(), SensitivityAtSpecificity(0.95)]

    # Get the model
    if m_mode:
        input_shape = [cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0], cfg['PREPROCESS']['PARAMS']['M_MODE_WIDTH'], 3]
    elif flow == 'Yes':
        input_shape = [cfg['PREPROCESS']['PARAMS']['WINDOW']] + cfg['PREPROCESS']['PARAMS']['IMG_SIZE'] + [2]
    else:
        input_shape = [cfg['PREPROCESS']['PARAMS']['WINDOW']] + cfg['PREPROCESS']['PARAMS']['IMG_SIZE'] + [3]
    if flow == 'Both' and not m_mode:
        input_shape = [input_shape]
        input_shape.append([cfg['PREPROCESS']['PARAMS']['WINDOW']] + cfg['PREPROCESS']['PARAMS']['IMG_SIZE'] + [2])

    model = model_def_fn(hparams, input_shape, metrics, counts)

    # Refresh the TensorBoard directory
    tensorboard_path = cfg['TRAIN']['PATHS']['TENSORBOARD']
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Log metrics
    if log_name is None:
        log_dir = cfg['TRAIN']['PATHS']['TENSORBOARD'] + time
    else:
        log_dir = cfg['TRAIN']['PATHS']['TENSORBOARD'] + log_name + '_' + time

    # uncomment line below to include tensorboard profiler in callbacks
    # basic_call = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=1)
    basic_call = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Learning rate scheduler & logging LR
    writer1 = tf.summary.create_file_writer(log_dir + '/train')

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
        with writer1.as_default():  # Write LR scalar to log directory
            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Creating a ModelCheckpoint for saving the model
    model_out_dir += time
    save_cp = ModelCheckpoint(model_out_dir, save_best_only=cfg['TRAIN']['SAVE_BEST_ONLY'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min',
                                   restore_best_weights=True)

    # Log model params to tensorboard
    writer2 = tf.summary.create_file_writer(log_dir + '/test')
    if log_dir is not None:
        log_train_params(writer1, hparams)

    # Train and save the model
    epochs = cfg['TRAIN']['PARAMS']['EPOCHS']
    history = None
    err = False
    try:
        if hparams['CLASS_WEIGHTS']:
            history = model.fit(train_set, epochs=epochs, validation_data=val_set, class_weight=class_weight,
                                callbacks=[save_cp, basic_call, early_stopping, lr_callback], verbose=1)
        else:
            history = model.fit(train_set, epochs=epochs, validation_data=val_set,
                                callbacks=[save_cp, basic_call, early_stopping, lr_callback], verbose=1)
    except tf.errors.InvalidArgumentError as e:
        logging.warning('An exception occurred: {}'.format(e))
        err = True

    # Log early stopping to tensorboard
    if history is not None:
        if len(history.epoch) < epochs:
            with writer2.as_default():
                tf.summary.text(name='Early Stopping', data=tf.convert_to_tensor('Training stopped early'), step=0)

    test_metrics = {}
    if cfg['TRAIN']['SPLITS']['TEST'] > 0:
        # Run the model on the test set and print the resulting performance metrics.
        test_results = model.evaluate(test_set, verbose=1)
        test_summary_str = [['**Metric**', '**Value**']]
        for metric, value in zip(model.metrics_names, test_results):
            test_metrics[metric] = value
            test_summary_str.append([metric, str(value)])
        if log_dir is not None:
            log_test_results(model, test_set, test_df, test_metrics, writer2)
            log_miniclip_test_results(model, test_set, test_df, writer2)
    else:
        test_results = model.evaluate(val_set, verbose=1)
        test_summary_str = [['**Metric**', '**Value**']]
        for metric, value in zip(model.metrics_names, test_results):
            test_metrics[metric] = value
            test_summary_str.append([metric, str(value)])
        if log_dir is not None:
            log_test_results(model, val_set, val_df, test_metrics, writer2)
            log_miniclip_test_results(model, val_set, val_df, writer2)

    # Adding error indicator, helpful for hparam search or cross val experiment
    test_metrics['err'] = err
    return model, test_metrics, test_set


def partition_k_fold(df):
    '''
    Partitions dataset into k folds by patient
    :param df: DataFrame (containing mini-clips, patient ids, npz paths, and labels) to be partitioned
    '''

    n_folds = cfg['CROSS_VAL']['N_FOLDS']

    # get unique patients
    all_pts = df['patient_id'].unique()
    val_split = 1.0 / n_folds
    cfg['TRAIN']['SPLITS']['VAL'] = val_split
    cfg['TRAIN']['SPLITS']['TEST'] = 0.
    pt_k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg['TRAIN']['SPLITS']['RANDOM_SEED'])

    labels = []
    for patient in all_pts:
        labels.append(df[df['patient_id'] == patient]['label'].iloc[0])
    return pt_k_fold, all_pts, labels


def save_hparam_search_results(init_dict, score, objective_metric, hparam_names, hparams, model_name, cur_datetime,
                               metric_names, metric_vals):
    '''
    Saves the results of a hyperparameter search in a table and a partial dependence plot
    :param init_dict: Initial results dictionary (hyperparameter names for keys and empty lists as values)
    :param score: Objective score for current iteration
    :param objective_metric: Name of the objective metric
    :param hparam_names: list of hyperparameter names
    :param hparams: Hyperparameter value dictionary
    :param model_name: Name of model
    :param cur_datetime: String representation of current date and time
    :return: Results dictionary
    '''

    # Create table to detail results
    results_path = cfg['HPARAM_SEARCH']['PATH'] + 'hparam_search_' + model_name + \
                   cur_datetime + '.csv'
    if os.path.exists(results_path):
        results = pd.read_csv(results_path).to_dict(orient='list')
    else:
        results = init_dict
    trial_idx = len(results['Trial'])
    results['Trial'].append(str(trial_idx))
    results[objective_metric].append(1.0 - score)
    for hparam_name in hparam_names:
        results[hparam_name].append(hparams[hparam_name])

    for i in range(len(metric_vals)):
        results[metric_names[i]].append(metric_vals[i])

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index_label=False, index=False)
    return results


def get_hparam_search_info():
    '''
    Gets hyperparameter types and ranges from config file
    '''
    objective_metric = cfg['HPARAM_SEARCH']['OBJECTIVE']
    model_name = cfg['TRAIN']['MODEL_DEF'].upper()
    results = {'Trial': [], objective_metric: []}
    dimensions = []
    default_params = []
    hparam_names = []
    for hparam_name in cfg['HPARAM_SEARCH'][model_name]:
        if cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'] is not None:
            if cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'set':
                dimensions.append(Categorical(categories=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'],
                                              name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'int_uniform':
                dimensions.append(Integer(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                          high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                          prior='uniform', name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_log':
                dimensions.append(Real(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                       high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                       prior='log-uniform', name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_uniform':
                dimensions.append(Real(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                       high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                       prior='uniform', name=hparam_name))
            default_params.append(cfg['HPARAM_SEARCH'][model_name][hparam_name])
            hparam_names.append(hparam_name)
            results[hparam_name] = []
    return results, dimensions, default_params, hparam_names


def bayesian_hparam_optimization(checkpoint=None, df=None, cross_val_runs=0):
    '''
    Conducts a Bayesian hyperparameter optimization
    :param checkpoint: string containing filename of checkpoint to load from previous search (.pkl file)
    :param df: DataFrame (mini-clips, patient ids, npz paths, and labels) to be partitioned for cross-validation
    :param cross_val_runs: number of runs for each hparam combination (on different validation folds)
    :return: Dict of hyperparameters deemed optimal
    '''
    model_name = cfg['TRAIN']['MODEL_DEF'].upper()
    cur_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    objective_metric = cfg['HPARAM_SEARCH']['OBJECTIVE']
    results, dimensions, default_params, hparam_names = get_hparam_search_info()
    print("Hyperparameter list: {}".format(hparam_names))

    metric_names = cfg['HPARAM_SEARCH']['METRICS']
    for m in metric_names:
        results[m] = []
    init_results = results

    # partition dataset for cross validation
    if cross_val_runs > 0 and df is not None:
        pt_k_fold, all_pts, labels = partition_k_fold(df)

        partition_path = os.path.join(cfg['CROSS_VAL']['PARTITIONS'], 'kfold' + cur_datetime)
        if not os.path.exists(partition_path):
            os.makedirs(partition_path)

        cur_fold = 0
        train_dfs = []
        val_dfs = []
        for train_index, val_index in pt_k_fold.split(all_pts, labels):
            # Partition into training and validation sets for this fold. Save the partitions.
            train_pts = all_pts[train_index]
            val_pts = all_pts[val_index]
            train_df = df[df['patient_id'].isin(train_pts)]
            val_df = df[df['patient_id'].isin(val_pts)]

            train_dfs.append(train_df)
            val_dfs.append(val_df)

            train_df.to_csv(os.path.join(partition_path, 'fold_' + str(cur_fold) + '_train_set.csv'), index=False)
            val_df.to_csv(os.path.join(partition_path, 'fold_' + str(cur_fold) + '_val_set.csv'), index=False)
            cur_fold += 1
            if cur_fold >= cross_val_runs:
                break

    def cross_val_objective(vals):
        hparams = dict(zip(hparam_names, vals))
        for hparam in cfg['TRAIN']['PARAMS'][model_name]:
            if hparam not in hparams:
                hparams[hparam] = cfg['TRAIN']['PARAMS'][model_name][hparam]  # Add hyperparameters being held constant
        print('HPARAM VALUES: ', hparams)
        # run multiple runs for this hyperparameter combination on different data partitions
        all_scores = []
        all_metrics = []
        for i in range(cross_val_runs):
            _, test_metrics, _ = train_model(hparams=hparams, train_df=train_dfs[i], val_df=val_dfs[i], log_name=str(i))
            if objective_metric == 'weighted_spec_recall':
                cur_score = 1.5 * (1. - test_metrics['specificity']) + (1. - test_metrics['recall'])
            else:
                cur_score = 1. - test_metrics[objective_metric]
            cur_metrics = []
            for metric in metric_names:
                cur_metrics.append(test_metrics[metric])
            all_scores.append(cur_score)
            all_metrics.append(cur_metrics)
            gc.collect()
            tf.keras.backend.clear_session()
        score = np.mean(all_scores)
        metric_vals = np.mean(all_metrics, axis=0)
        save_hparam_search_results(init_results, score, objective_metric, hparam_names, hparams, model_name,
                                   cur_datetime, metric_names, metric_vals)
        return score

    def objective(vals):
        hparams = dict(zip(hparam_names, vals))
        for hparam in cfg['TRAIN']['PARAMS'][model_name]:
            if hparam not in hparams:
                hparams[hparam] = cfg['TRAIN']['PARAMS'][model_name][hparam]  # Add hyperparameters being held constant
        print('HPARAM VALUES: ', hparams)

        _, test_metrics, _ = train_model(hparams=hparams)
        metric_vals = []
        for metric in metric_names:
            metric_vals.append(test_metrics[metric])
        if objective_metric == 'weighted_spec_recall':
            score = 1.5 * (1. - test_metrics['specificity']) + (1. - test_metrics['recall'])
        else:
            score = 1. - test_metrics[objective_metric]
        save_hparam_search_results(init_results, score, objective_metric, hparam_names, hparams, model_name,
                                   cur_datetime, metric_names, metric_vals)
        gc.collect()
        tf.keras.backend.clear_session()
        return score  # We aim to minimize error

    objective_func = cross_val_objective if cross_val_runs > 0 and df is not None else objective
    result_path = cfg['HPARAM_SEARCH']['PATH'] + 'hparam_search_' + model_name + cur_datetime + '.pkl'
    checkpoint_saver = CheckpointSaver(result_path)

    # whether to load from checkpoint
    if checkpoint:
        print('here')
        checkpoint_res = load(os.path.join(cfg['HPARAM_SEARCH']['PATH'], checkpoint))
        x0 = checkpoint_res.x_iters
        y0 = checkpoint_res.func_vals
        search_results = gp_minimize(func=objective_func, dimensions=dimensions, x0=x0, y0=y0, n_initial_points=-len(x0),
                                     acq_func='EI', n_calls=cfg['HPARAM_SEARCH']['N_EVALS'] - len(x0),
                                     callback=[checkpoint_saver], verbose=True)
    else:
        search_results = gp_minimize(func=objective_func, dimensions=dimensions, acq_func='EI',
                                     n_calls=cfg['HPARAM_SEARCH']['N_EVALS'], callback=[checkpoint_saver],
                                     verbose=True)
    print("Results of hyperparameter search: {}".format(search_results))
    plot_bayesian_hparam_opt(model_name, hparam_names, search_results, save_fig=True)
    return search_results


def cross_validation(df=None, hparams=None):
    '''
    Perform k-fold cross-validation. Results are saved in CSV format.
    :param df: A DataFrame consisting of the entire dataset of LUS frames
    :param save_weights: Flag indicating whether to save model weights
    :return DataFrame of metrics
    '''

    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    csv_path = os.path.join(os.getcwd(), 'data/', cfg['PREPROCESS']['PATHS']['CSVS_OUTPUT'])
    npz_dir = cfg['PREPROCESS']['PATHS']['NPZ']
    n_folds = cfg['CROSS_VAL']['N_FOLDS']

    metrics = cfg['CROSS_VAL']['METRICS']
    metrics_df = pd.DataFrame(np.zeros((n_folds + 2, len(metrics) + 1)), columns=['Fold'] + metrics)
    metrics_df['Fold'] = list(range(n_folds)) + ['mean', 'std']

    model_name = cfg['TRAIN']['MODEL_DEF'].lower()
    hparams = cfg['TRAIN']['PARAMS'][model_name.upper()] if hparams is None else hparams

    partition_path = os.path.join(cfg['CROSS_VAL']['PARTITIONS'], 'kfold' + cur_date)
    if not os.path.exists(partition_path):
        os.makedirs(partition_path)

    # partition data
    pt_k_fold, all_pts, labels = partition_k_fold(df)

    # Train a model n_folds times with different folds
    cur_fold = 0
    for train_index, val_index in pt_k_fold.split(all_pts, labels):
        print('FITTING MODEL FOR FOLD ' + str(cur_fold))

        # Partition into training and validation sets for this fold. Save the partitions.
        train_pts = all_pts[train_index]
        val_pts = all_pts[val_index]
        train_df = df[df['patient_id'].isin(train_pts)]
        val_df = df[df['patient_id'].isin(val_pts)]

        train_df.to_csv(os.path.join(partition_path, 'fold_' + str(cur_fold) + '_train_set.csv'), index=False)
        val_df.to_csv(os.path.join(partition_path, 'fold_' + str(cur_fold) + '_val_set.csv'), index=False)
        print('TRAIN/VAL SPLIT: [{}, {}] mini-clips, [{}, {}] patients'
              .format(train_df.shape[0], val_df.shape[0], train_pts.shape[0], val_pts.shape[0]))

        # Train the model and evaluate performance on test set
        model, val_metrics, _ = train_model(hparams=hparams, train_df=train_df, val_df=val_df)

        for metric in val_metrics:
            if metric in metrics_df.columns:
                metrics_df.loc[cur_fold, metric] = val_metrics[metric]
        cur_fold += 1
        gc.collect()
        tf.keras.backend.clear_session()
        del model

    # Record mean and standard deviation of test set results
    for metric in metrics:
        metrics_df[metric][n_folds] = metrics_df[metric][0:-2].mean()
        metrics_df[metric][n_folds + 1] = metrics_df[metric][0:-2].std()

    # Save results
    file_path = os.path.join(cfg['TRAIN']['PATHS']['EXPERIMENTS'], 'cross_val_' + model_name + cur_date + '.csv')
    metrics_df.to_csv(file_path, columns=metrics_df.columns, index_label=False, index=False)
    return metrics_df


def train_experiment(experiment='single_train', checkpoint=None):
    '''
    Run a specific type of training experiment (single, hparam search, hparam search with cross val, cross val)
    :param experiment: type of experiment
    :param checkpoint: name of checkpoint to load for hparam search
    '''
    model_def_str = cfg['TRAIN']['MODEL_DEF']
    if experiment == 'single_train':
        train_model(hparams=cfg['TRAIN']['PARAMS'][model_def_str.upper()])
    elif experiment == 'hparam_search':
        if cfg['HPARAM_SEARCH']['LOAD_CHECKPOINT']:
            checkpoint = cfg['HPARAM_SEARCH']['CHECKPOINT']
        bayesian_hparam_optimization(checkpoint=checkpoint)
    elif experiment == 'hparam_cross_val_search':
        if cfg['HPARAM_SEARCH']['LOAD_CHECKPOINT']:
            checkpoint = cfg['HPARAM_SEARCH']['CHECKPOINT']
        train_df = pd.read_csv(os.path.join(cfg['TRAIN']['PATHS']['CSVS'], 'train.csv'))
        val_df = pd.read_csv(os.path.join(cfg['TRAIN']['PATHS']['CSVS'], 'val.csv'))
        df = pd.concat([train_df, val_df]).reset_index(drop=True)
        bayesian_hparam_optimization(checkpoint=checkpoint, df=df, cross_val_runs=cfg['HPARAM_SEARCH']['CROSS_VAL_RUNS'])
    elif experiment == 'cross_val':
        train_df = pd.read_csv(os.path.join(cfg['TRAIN']['PATHS']['CSVS'], 'train.csv'))
        val_df = pd.read_csv(os.path.join(cfg['TRAIN']['PATHS']['CSVS'], 'val.csv'))
        df = pd.concat([train_df, val_df]).reset_index(drop=True)
        cross_validation(df, hparams=cfg['TRAIN']['PARAMS'][model_def_str.upper()])
    return


if __name__ == '__main__':
    train_experiment(cfg['TRAIN']['EXPERIMENT_TYPE'])