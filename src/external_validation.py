import gc
import glob
import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.losses import SigmoidFocalCrossEntropy, sigmoid_focal_crossentropy
from models.models import get_model
from predict import predict_set
from src.custom.metrics import Specificity
from src.data.sql_utils import add_date_to_filename
from src.data.results.finetune_results_utils import finetune_results_to_csv, plot_finetune_results
from src.data.utils import refresh_folder
from src.preprocessor import MModePreprocessor
from train import log_train_params

# cfg is the sub-dictionary of the config file pertaining to fine-tuning experiments. cfg_full is the full config file.
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))['EXTERNAL_VAL']
cfg_full = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))


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
    # Set up the folds csv.
    folds_csv = pd.DataFrame([])
    fold_nums = []
    i = 1
    for fold in folds:
        folds_csv = pd.concat([folds_csv, fold], ignore_index=True)
        fold_nums.extend([i] * len(fold))
        i += 1

    # Append to the folds df a column for specifying fold number.
    fold_nums = pd.DataFrame(fold_nums, columns=['fold_number']).reset_index(drop=True)
    folds_csv = folds_csv.reset_index(drop=True)
    folds_csv.columns = ['id']
    folds_csv = pd.concat([fold_nums, folds_csv], axis=1)
    folds_csv.to_csv(file_path)


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
    for center in cfg['LOCATIONS']:
        center_split_path = os.path.join(os.getcwd() + cfg['PATHS']['CSVS_SPLIT'], center)
        for csv in ['train.csv', 'val.csv']:
            ext_dfs.append(pd.read_csv(os.path.join(center_split_path, csv)))
    ext_df = pd.concat(ext_dfs, ignore_index=True)
    labelled_ext_df_path = os.path.join(os.getcwd() + cfg['PATHS']['CSV_OUT'], 'labelled_external_dfs/')
    if cfg['REFRESH_FOLDERS']:
        refresh_folder(labelled_ext_df_path)
    labelled_ext_df_path += add_date_to_filename('labelled_ext') + '.csv'
    ext_df.to_csv(labelled_ext_df_path)

    return ext_df


def get_eval_results(model_path, test_df, metrics, save_pred_labels=True, verbose=True):
    '''
    Returns a df containing evaluation results of the model stored at model_path on test_set, using specified metrics.
    :param model_path: Absolute path to model.
    :param test_df: DataFrame on which to evaluate the model.
    :param metrics: List of metrics for which to report evaluation results.
    :param verbose: Boolean. If True, prints out test results in list and df form (default value is True).
    '''
    # Load and compile model from model_path.
    model = load_model(model_path, compile=False)

    # Read values from the config file.
    model_config = cfg_full['TRAIN']['PARAMS']['EFFICIENTNET']
    lr = cfg['FINETUNE']['LR']
    optimizer = Adam(learning_rate=lr)

    # Compile the model using specified list of metrics.
    model.compile(loss=SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO,
                                                alpha=model_config['ALPHA'],
                                                gamma=model_config['GAMMA']),
                  optimizer=optimizer, metrics=metrics)
    pred_labels, probs = predict_set(model, test_df)

    # Saving prediction labels and probabilities.
    pred_label_df = None
    if save_pred_labels:
        pred_labels_col = pd.DataFrame(pred_labels, columns=['pred_label']).reset_index(drop=True)
        probs_col = pd.DataFrame(probs, columns=['prob']).reset_index(drop=True)
        ids = pd.DataFrame(test_df['id'], columns=['id']).reset_index(drop=True)
        pred_label_df = pd.concat([ids, pred_labels_col, probs_col], axis=1)

    conf_mtrx = confusion_matrix(y_true=test_df['label'], y_pred=np.rint(probs))

    results = []
    metric_names = []

    for metric in metrics:
        # Probabilities are stored as floats, while Accuracy metric compares integer values for labels and
        # probs. Round probabilities to the closer value of 0 or 1.
        if metric.name == 'Accuracy':
            probs = np.rint(probs)
        metric.update_state(test_df['label'], probs)
        results += [metric.result().numpy()]
        metric_names += [metric.name]

    # Flatten the confusion matrix into a list storing: [tn, fp, fn, tp].
    results += [conf_mtrx.ravel()]
    metric_names += ['confusion_matrix']

    # Create df storing evaluation results.
    res_df = pd.DataFrame([results], columns=metric_names)

    # Record loss value on external test set.
    true_labels = np.array(test_df['label']).astype(np.float)
    loss_value = np.array(sigmoid_focal_crossentropy(y_true=true_labels, y_pred=probs))
    loss_value = loss_value / len(probs)
    loss_value = pd.DataFrame([loss_value], columns=['loss'])
    res_df = pd.concat([loss_value, res_df], axis=1)

    if save_pred_labels == 1:
        return res_df, pred_label_df
    return res_df, None


def setup_experiment(experiment_path=None):
    '''
    Creates a new experiment folder which contains one subdirectory for each trial in the experiment. The newly
    created experiment folder is labelled with a Unix timestamp. Returns the path to the newly created experiment.
    :param experiment_path: Absolute path to folder storing experiments.
    :returns: Absolute path to newly created experiment.
    '''
    if experiment_path is None:
        experiment_folder = os.getcwd() + cfg['PATHS']['EXPERIMENTS']
        experiment_path = os.path.join(experiment_folder, add_date_to_filename('experiment'))

    os.makedirs(experiment_path)

    # Create trial folders
    for trial_num in range(cfg['NUM_TRIALS']):
        trial_path = os.path.join(experiment_path, 'trial_' + str(trial_num + 1))
        os.makedirs(trial_path)

    return experiment_path


def upsample_absent_sliding_data(train_df, upsample_df, prop, absent_pct, method='gaussian_noise'):
    '''
    Upsamples the absent lung sliding M-Modes in the given training dataset, train_df.
    :param train_df: Training dataset for which to upsample minority.
    :param upsample_df: DataFrame indicating which M-Modes have already been upsampled.
    :param prop: Proportion of external dataset used for training.
    :param absent_pct: Percentage of training dataset that should be from the absent lung sliding class after the
    upsampling.
    :returns: Returns two pd.DataFrames: upsampled train set, and updated upsample set.
    '''
    sliding_df = train_df[train_df['label'] == 1]
    no_sliding_df = train_df[train_df['label'] == 0]

    if len(no_sliding_df) / len(train_df) < absent_pct:
        # Upsample
        print('No sliding before upsampling:', len(no_sliding_df))
        print('Upsampling...')
        no_sliding_df = no_sliding_df.sample(frac=1).reset_index(drop=True)
        num_upsampled = int(absent_pct * (len(sliding_df) / (1 - absent_pct)))

        # Clips in no_sliding_df that are not in upsample_df
        clips_for_upsample = no_sliding_df[~no_sliding_df['id'].isin(list(upsample_df['miniclip_id']))].copy()
        clips_for_upsample = clips_for_upsample.reset_index(drop=True)
        print('Number of clips available for upsampling:', len(clips_for_upsample))
        print('Number of clips needed:', num_upsampled - len(no_sliding_df))

        new_upsample_clips = clips_for_upsample.iloc[:(num_upsampled - len(no_sliding_df))]

        # If run out of clips that have not been upsampled already, can upsample remaining clips needed from clips
        # already upsampled
        n_extra_upsample = (num_upsampled - len(no_sliding_df)) - len(clips_for_upsample)
        if n_extra_upsample > 0:
            temp_df = no_sliding_df.copy().sample(frac=1).reset_index(drop=True)
            extra_upsample_clips = temp_df.iloc[:n_extra_upsample]
            # extra_upsample_clips.rename(columns={'id': 'miniclip_id'}, inplace=True)
            new_upsample_clips = pd.concat([new_upsample_clips, extra_upsample_clips]).reset_index(drop=True)

        print('Number of clips upsampled:', len(new_upsample_clips))
        no_sliding_df = pd.concat([no_sliding_df, new_upsample_clips])
        print('No sliding after upsampling:', len(no_sliding_df))
        new_train_df = pd.concat([sliding_df, no_sliding_df])

        # Update the upsample df.
        ext_data_props = pd.DataFrame([prop] * len(new_upsample_clips), columns=['ext_data_prop']).reset_index(
            drop=True)
        new_upsample_clips.rename(columns={'id': 'miniclip_id'}, inplace=True)
        upsample_df_addition = pd.concat([ext_data_props, new_upsample_clips[['miniclip_id']]], axis=1)
        new_upsample_df = pd.concat([upsample_df, upsample_df_addition])

        print('Sliding: {}'.format(len(sliding_df) / len(new_train_df)))
        print('No Sliding: {}'.format(len(no_sliding_df) / len(new_train_df)))

        return new_train_df, new_upsample_df

    else:
        return train_df, upsample_df


    # # If there are upsampled mini-clips already
    # if len(upsample_df) != 0:
    #     no_upsampled_train_df = train_df[~train_df['id'].isin(upsample_df['miniclip_id'])].copy()
    #     print('Number of clips that can be upsampled:', len(no_upsampled_train_df))
    # else:
    #     no_upsampled_train_df = train_df.copy()
    #
    # # Get sliding and no-sliding subsets of train_df.
    # sliding_df = no_upsampled_train_df[no_upsampled_train_df['label'] == 1]
    # no_sliding_df = no_upsampled_train_df[no_upsampled_train_df['label'] == 0]
    #
    # sliding_df = sliding_df.reset_index(drop=True)
    #
    # # Shuffle no_sliding_df. We do this to facilitate random duplicate up-sampling from the no-sliding dataframe.
    # # I.e we draw miniclips to duplicate from a shuffled no-sliding df.
    #
    #
    # # number of absent sliding clips after upsampling.
    #
    #

    #
    # no_sliding_df = pd.concat([no_sliding_df, upsample_set])
    # new_train_df = pd.concat([sliding_df, no_sliding_df])
    # print('Sliding: {}'.format(len(sliding_df) / len(new_train_df)))
    # print('No Sliding: {}'.format(len(no_sliding_df) / len(new_train_df)))
    #
    # return new_train_df, new_upsampled_df


# TAAFT stands for: Threshold-Aware Accumulative Fine-Tuner
class TAAFT:
    '''
    The TAAFT class encapsulates the implementation of the fine-tuning for generalizability experiments.
    Handles fold sampling and fine-tuning trials.
    '''
    def __init__(self, mini_clip_df, k):
        '''
        :param clip_df: DataFrame containing all clip data from external centers.
        :param k: Number of folds in which to sample data in df by patient id. Note that these folds, while sampled by
        patient id, are stored by clip id.
        '''
        print("CREATING A NEW TAAFT (k = {})".format(k))
        self.k = k
        self.mini_clip_df = mini_clip_df
        np.random.seed(cfg['FOLD_SAMPLE']['SEED'])

    def sample_folds(self, trial_folder):
        '''
        Samples external data in self.clip_df into self.k folds. Preserves ratio of positive to negative classes in each
        fold. Folds are saved in a .txt file, where each fold is identified with an integer index, followed by each clip
        id in the fold on a new line.
        :param trial_folder: Absolute path to folder in which to place patient folds. Corresponds to a particular trial.
        '''

        sliding_df = self.mini_clip_df[self.mini_clip_df['label'] == 1]
        no_sliding_df = self.mini_clip_df[self.mini_clip_df['label'] == 0]

        # Get unique patient ids.
        sliding_ids = sliding_df['patient_id'].unique()
        no_sliding_ids = no_sliding_df['patient_id'].unique()

        print("{} unique sliding patient ids found.".format(len(sliding_ids)))
        print("{} unique no-sliding patient ids found.\n".format(len(no_sliding_ids)))

        # Shuffle the patient ids.
        np.random.shuffle(sliding_ids)
        np.random.shuffle(no_sliding_ids)

        trial_name = trial_folder.split('\\')[-1]
        folds_path = os.path.join(trial_folder, add_date_to_filename('folds_' + trial_name) + '.csv')

        # Make the file that stores ids by fold if it doesn't already exist.
        # Clear the file if user intends to overwrite its contents.
        overwrite = cfg['PARAMS']['OVERWRITE_FOLDS']
        if not os.path.exists(folds_path):
            open(folds_path, "x").close()
        elif overwrite:
            open(folds_path, "w").close()

        # Keep track of the desired sizes for each fold. Since not all folds will be of the same size in terms
        # of patient ids, this eliminates the need to dynamically compute each fold size.
        n_folds = cfg['FOLD_SAMPLE']['NUM_FOLDS']
        fold_sizes = [len(sliding_ids) // n_folds] * n_folds
        fold_sizes[-1] += len(sliding_ids) % n_folds

        folds = []
        sliding_count = [0] * n_folds
        no_sliding_count = [0] * n_folds

        folds_to_supply = []

        for i in range(n_folds):
            fold_end = min(fold_sizes[i], len(sliding_ids))
            cur_ids = sliding_ids[:fold_end]
            sliding_ids = sliding_ids[fold_sizes[i]:]
            cur_fold_sliding = sliding_df[sliding_df['patient_id'].isin(cur_ids)]['id']
            cur_fold_no_sliding = no_sliding_df[no_sliding_df['patient_id'].isin(cur_ids)]['id']
            sliding_count[i] += len(cur_fold_sliding)
            no_sliding_count[i] += len(cur_fold_no_sliding)
            if len(cur_fold_no_sliding) == 0:
                folds_to_supply.append(i)
            no_sliding_df = no_sliding_df[~no_sliding_df['patient_id'].isin(cur_ids)]
            folds.append(pd.concat([cur_fold_sliding, cur_fold_no_sliding]))

        no_sliding_ids = no_sliding_df['patient_id'].values
        folds_with_all_sliding = len(folds_to_supply) if len(folds_to_supply) > 0 else n_folds
        no_sliding_to_add = [len(no_sliding_ids) // folds_with_all_sliding] * folds_with_all_sliding
        for i in range(len(no_sliding_ids) % folds_with_all_sliding):
            no_sliding_to_add[i] += 1

        for i, j in zip(no_sliding_to_add, folds_to_supply):
            folds[j] = pd.concat([folds[j], no_sliding_df.head(i)])
            no_sliding_df = no_sliding_df.tail(len(no_sliding_df) - i)

        ratios = []
        for i in range(n_folds):
            ratio = no_sliding_count[i] / sliding_count[i]
            ratios.append(ratio)
            print('Fold {}: {} sliding, {} no sliding. Absent to present ratio: {}'
                  .format(i + 1, sliding_count[i], no_sliding_count[i], ratio))

        print('Average absent to present ratio: {}'.format(sum(ratios) / n_folds))

        # Write the folds to the fold file:
        if cfg['FOLD_SAMPLE']['OVERWRITE_FOLDER']:
            refresh_folder(os.getcwd() + cfg['PATHS']['FOLDS'])
        write_folds_to_csv(folds, folds_path)

    def finetune_single_trial(self, trial_folder, hparams=None, lazy=True, upsample=True):
        '''
        Performs a single fine-tuning trial. Fine-tunes a model on external clip data, repeatedly augmenting external
        data slices to a training set until all external data has been used for training.
        :param trial_folder: Absolute path to folder in which to place metrics for current trial.
        :param hparams: Specifies the set of hyperparameters for fine-tuning the model.
        :param lazy: When True, trial will terminate once model performance exceeds minimum thresholds.
        :param upsample: When True, upsamples absent sliding mini-clips in each training set during fine-tuning.
        '''

        original_model_path = os.path.join(os.getcwd(), cfg_full['PREDICT']['MODEL'])
        metrics = [Recall(name='sensitivity'), Specificity(name='specificity')]

        # Set up result directories for given trial; read values from config file.
        model_out_dir = os.path.join(trial_folder, 'models')
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)

        # Set up the labelled external dataframe for augmenting the training data.
        ext_df = self.mini_clip_df

        # This df will store trial results.
        trial_results_df = pd.DataFrame([])

        # Get results for original model before fine-tuning.
        res_df, _ = get_eval_results(original_model_path, ext_df, metrics)
        trial_results_df = pd.concat([trial_results_df, res_df])
        original_model = load_model(original_model_path, compile=False)
        original_model.save(os.path.join(model_out_dir, 'model_0.0_percent_train'))

        model_name = cfg_full['TRAIN']['MODEL_DEF'].lower()
        hparams = cfg_full['TRAIN']['PARAMS'][model_name.upper()] if hparams is None else hparams

        model_def = cfg_full['TRAIN']['MODEL_DEF'].upper()
        model_config = cfg_full['TRAIN']['PARAMS'][model_def]

        # Retrieve patient folds (stored by clip ids).
        filename = glob.glob(os.path.join(trial_folder, 'folds*.csv'))[0]
        folds = pd.read_csv(filename)

        train_df = pd.DataFrame([])

        # If absent sliding mini-clips are to be upsampled, create a df for saving these upsampled mini-clips.
        upsample_df = pd.DataFrame([], columns=['ext_data_prop', 'miniclip_id'])
        upsample_miniclips_path = os.path.join(trial_folder, 'upsample_miniclips.csv')

        min_test_size = int(len(ext_df) * cfg['MIN_TEST_SIZE'])

        cur_fold_num = 0
        props = [0]
        while len(ext_df) > min_test_size:
            print('Fine-tuning: {} of {} external data slices...'
                  .format(cur_fold_num + 1, cfg['FOLD_SAMPLE']['NUM_FOLDS']))
            fold_to_add = folds[folds['fold_number'] == cur_fold_num + 1]
            train_df = pd.concat([train_df, ext_df[ext_df['id'].isin(fold_to_add['id'])]])
            ext_df = ext_df[~ext_df['id'].isin(fold_to_add['id'])]

            # Model starts out as the original model, but then gets fine-tuned.
            model = original_model
            metrics = [Recall(name='sensitivity'), Specificity(name='specificity')]

            # Freeze some layers.
            freeze_cutoff = -1 if model_config['BLOCKS_FROZEN'] == 0 else model_config['BLOCKS_FROZEN']
            for i in range(len(model.layers)):
                if i <= freeze_cutoff:
                    model.layers[i].trainable = False
                # Freeze all batch norm layers
                elif ('batch' in model.layers[i].name) or ('bn' in model.layers[i].name):
                    model.layers[i].trainable = False

            # Create training and validation sets for fine-tuning.
            clips_not_upsampled = train_df[~train_df['id'].isin(upsample_df['miniclip_id'])]
            val_num = int(len(clips_not_upsampled) * cfg['FINETUNE']['VAL_SPLIT'])
            sub_val_df = clips_not_upsampled.tail(val_num)
            sub_train_df = train_df[~train_df['id'].isin(list(sub_val_df['id']))]
            print(len(sub_val_df[sub_val_df['label'] == 0]), len(sub_train_df[sub_train_df['label'] == 0]))
            print(len(sub_val_df[sub_val_df['label'] == 1]), len(sub_train_df[sub_train_df['label'] == 1]))
            print(len(set(sub_train_df['id']) - set(sub_val_df['id'])) == len(sub_train_df['id'].unique()))
            print(len(set(sub_val_df['id']) - set(sub_train_df['id'])) == len(sub_val_df['id'].unique()))

            if upsample:
                # cur_ext_prop is the current external data proportion used for training
                cur_ext_prop = (cur_fold_num + 1) / cfg['FOLD_SAMPLE']['NUM_FOLDS']
                sub_train_df, upsample_df = upsample_absent_sliding_data(sub_train_df, upsample_df, cur_ext_prop, 0.25)
                train_df = pd.concat([sub_train_df, sub_val_df])

                # Update the upsampled mini-clips .csv file.
                if os.path.exists(upsample_miniclips_path):
                    os.remove(upsample_miniclips_path)
                upsample_df.to_csv(upsample_miniclips_path)

            # Shuffle the training data.
            train_df.sample(frac=1)

            print('Train: {}\nTest: {}'.format(len(train_df), len(ext_df)))

            # ========== The following code was borrowed from src/models/models.py. ====================================
            # Fine-tune the model
            model_def_fn, preprocessing_fn = get_model(cfg['FINETUNE']['MODEL_DEF'])

            # Prepare the training and validation sets from the dataframes.
            train_set = tf.data.Dataset.from_tensor_slices(
                (sub_train_df['filename'].tolist(), sub_train_df['label'].tolist()))
            val_set = tf.data.Dataset.from_tensor_slices(
                (sub_val_df['filename'].tolist(), sub_val_df['label'].tolist()))
            test_set = tf.data.Dataset.from_tensor_slices((ext_df['filename'].tolist(), ext_df['label'].tolist()))

            preprocessor = MModePreprocessor(preprocessing_fn)

            # Define the preprocessing pipelines for train, test and validation
            train_set = preprocessor.prepare(train_set, sub_train_df, shuffle=True, augment=True)
            val_set = preprocessor.prepare(val_set, sub_val_df, shuffle=False, augment=False)
            test_set = preprocessor.prepare(test_set, ext_df, shuffle=False, augment=False)

            # Create dictionary for class weights like in src/train.py.
            num_no_sliding = len(train_df[train_df['label'] == 0])
            num_sliding = len(train_df[train_df['label'] == 1])
            total = num_no_sliding + num_sliding
            weight_for_0 = (1 / num_no_sliding) * (total / 2.0)
            weight_for_1 = (1 / num_sliding) * (total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}

            # Refresh the TensorBoard directory
            tensorboard_path = os.getcwd() + os.path.join(cfg_full['TRAIN']['PATHS']['TENSORBOARD'], add_date_to_filename('log'))
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path)

            # uncomment line below to include tensorboard profiler in callbacks
            # basic_call = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=1)
            basic_call = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)

            # Learning rate scheduler & logging LR
            writer1 = tf.summary.create_file_writer(tensorboard_path + '/train')

            scheduler = make_scheduler(writer1, hparams)
            lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

            # Creating a ModelCheckpoint for saving the model
            save_cp = ModelCheckpoint(os.path.join(model_out_dir, 'model_'
                                                   + str((cur_fold_num + 1) / cfg['FOLD_SAMPLE']['NUM_FOLDS'])
                                                   + '_percent_train'),
                                      save_best_only=cfg_full['TRAIN']['SAVE_BEST_ONLY'])

            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg_full['TRAIN']['PATIENCE'],
                                           mode='min',
                                           restore_best_weights=True, min_delta=cfg['FINETUNE']['MIN_DELTA'])

            # Log model params to tensorboard
            writer2 = tf.summary.create_file_writer(tensorboard_path + '/test')
            if tensorboard_path is not None:
                log_train_params(writer1, hparams)

            lr = cfg['FINETUNE']['LR']

            optimizer = Adam(learning_rate=lr)
            model.compile(loss=SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO,
                                                        alpha=model_config['ALPHA'], gamma=model_config['GAMMA']),
                          optimizer=optimizer, metrics=metrics)

            # Train and save the model
            epochs = cfg['FINETUNE']['EPOCHS']
            history = model.fit(train_set, epochs=epochs, validation_data=val_set,
                                class_weight=class_weight,
                                callbacks=[save_cp, basic_call, early_stopping, lr_callback], verbose=1)

            # Log early stopping to tensorboard
            if history is not None:
                if len(history.epoch) < epochs:
                    with writer2.as_default():
                        tf.summary.text(name='Early Stopping', data=tf.convert_to_tensor('Training stopped early'),
                                        step=0)

            # Add evaluation results for the current model on the current test set to the trial results df.
            current_model_path = os.path.join(model_out_dir, os.listdir(model_out_dir)[-1])
            metrics = [Recall(name='sensitivity'), Specificity(name='specificity')]
            current_results, _ = get_eval_results(current_model_path, ext_df, metrics)
            trial_results_df = pd.concat([trial_results_df, current_results])

            # If the fine-tuned model now performs well on the test set, terminate the trial early.
            perf_check = check_performance(current_results)
            if perf_check and lazy:
                break

            gc.collect()
            tf.keras.backend.clear_session()
            del model

            # ==========================================================================================================

            # Update loop counters.
            cur_fold_num += 1
            props.append(cur_fold_num / cfg['FOLD_SAMPLE']['NUM_FOLDS'])

        print('EVALUATING PREVIOUS MODELS ON MINIMUM SIZED TEST SET...')

        # Set up column in trial results df to specify whether results are for fixed or variable sized test set.
        var_size_test_set = [1] * (cur_fold_num)

        # Add an extra 1 in the test set size column. This is for when we evaluated the original (non finetuned)
        # model on the external dataset.
        var_size_test_set.append(1)

        # For each previous model, get results on minimum (fixed) size test set.
        for prev_model in os.listdir(model_out_dir):
            prev_model_path = os.path.join(model_out_dir, prev_model)
            metrics = [Recall(name='sensitivity'), Specificity(name='specificity')]
            fixed_test_set_eval, _ = get_eval_results(prev_model_path, ext_df, metrics)
            trial_results_df = pd.concat([trial_results_df, fixed_test_set_eval])
            var_size_test_set.append(0)

        # Concatenate fixed/variable size test set column with the trial result df.
        var_size_test_set = pd.DataFrame(var_size_test_set, columns=['variable_sized_test_set']) \
            .reset_index(drop=True)
        props.extend(props)
        props = pd.DataFrame(props, columns=['ext_data_prop']).reset_index(drop=True)
        trial_results_df = trial_results_df.reset_index(drop=True)
        trial_results_df = pd.concat([var_size_test_set, props, trial_results_df], axis=1)
        trial_results_df.to_csv(os.path.join(trial_folder, 'results.csv'))


    def finetune_multiple_trials(self):
        '''
        Runs an experiment consisting of multiple trials. Saves results into .csv format and generates plots for the
        relevant metric results (by default: loss, sensitivity, and specificity) for each trial.
        '''
        experiment_path = setup_experiment()
        for trial in os.listdir(experiment_path):
            cur_trial_dir = os.path.join(experiment_path, trial)
            self.sample_folds(cur_trial_dir)
            self.finetune_single_trial(cur_trial_dir, lazy=cfg['FINETUNE']['LAZY'])

        # Save finetune results to combined result .csv file.
        # results_path = finetune_results_to_csv(experiment_path)
        # plot_finetune_results(results_path)


if __name__ == '__main__':
    # If the generalizability folder has not been created, do this first. Create the subdirectories for each of the
    # required data types (e.g. raw clips, .npzs, m-modes, etc.)
    for path in cfg['PATHS']:
        if not os.path.exists(os.getcwd() + cfg['PATHS'][path]):
            os.makedirs(os.getcwd() + cfg['PATHS'][path])

    external_df = get_external_clip_df()

    # Construct a TAAFT instance with the dataframe.
    taaft = TAAFT(external_df, cfg['FOLD_SAMPLE']['NUM_FOLDS'])
    taaft.finetune_multiple_trials()

