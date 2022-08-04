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


def write_folds_to_txt(folds, file_path):
    '''
    Saves clip id folds to .txt file in the following structure:
    -> An integer, n, which indicates the length of the current fold
    -> Followed by n lines, each containing one (unique) clip id
    :param folds: 2D list of strings. Each row represents a list of folds.
    :param file_path: Absolute path where folds should be stored.
    '''
    txt = open(file_path, "a")
    for i in range(len(folds)):
        txt.write(str(len(folds[i])) + '\n')
        for id in folds[i]:
            txt.write(str(id) + '\n')
    txt.close()


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
    csvs_dir = os.getcwd() + cfg['PATHS']['CSV_OUT']
    external_data = []

    # Combined clip info for all multi-center study locations.
    for center in cfg['LOCATIONS']:
        csv_folder = os.path.join(csvs_dir, center)
        for input_class in ['sliding', 'no_sliding']:
            csv_file = os.path.join(csv_folder, input_class + '.csv')
            external_data.append(pd.read_csv(csv_file))
    external_df = pd.concat(external_data)

    # Set up folder to store the external clip df.
    external_data_dir = os.path.join(os.getcwd() + cfg['PATHS']['CSV_OUT'], 'combined_external_data')
    if not os.path.exists(external_data_dir):
        os.makedirs(external_data_dir)
    if cfg['REFRESH_FOLDERS']:
        refresh_folder(external_data_dir)
    external_df_path = os.path.join(external_data_dir, add_date_to_filename('all_external_clips') + '.csv')

    # Save the external clip df as a .csv file.
    external_df.to_csv(external_df_path)
    print("All external clips consolidated into", external_df_path, '\n')

    return external_df


def get_eval_results(model_path, test_df, metrics, verbose=True):
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
    conf_mtrx = confusion_matrix(y_true=pred_labels, y_pred=np.rint(probs))

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

    return res_df


def setup_experiment(experiment_path=None):
    if experiment_path is None:
        experiment_folder = os.getcwd() + cfg['PATHS']['EXPERIMENTS']
        experiment_path = os.path.join(experiment_folder, add_date_to_filename('experiment'))

    os.makedirs(experiment_path)

    # Create trial folders
    for trial_num in range(cfg['NUM_TRIALS']):
        trial_path = os.path.join(experiment_path, 'trial_' + str(trial_num + 1))
        os.makedirs(trial_path)

    return experiment_path


# TAAFT stands for: Threshold-Aware Accumulative Fine-Tuner
class TAAFT:
    '''
    The TAAFT class encapsulates the implementation of the fine-tuning for generalizability experiments.
    Handles fold sampling and fine-tuning trials.
    '''
    def __init__(self, clip_df, k):
        '''
        :param clip_df: DataFrame containing all clip data from external centers.
        :param k: Number of folds in which to sample data in df by patient id. Note that these folds, while sampled by
        patient id, are stored by clip id.
        '''
        print("CREATING A NEW TAAFT (k = {})".format(k))
        self.k = k
        self.clip_df = clip_df
        np.random.seed(cfg['FOLD_SAMPLE']['SEED'])

    def sample_folds(self, trial_folder):
        '''
        Samples external data in self.clip_df into self.k folds. Preserves ratio of positive to negative classes in each
        fold. Folds are saved in a .txt file, where each fold is identified with an integer index, followed by each clip
        id in the fold on a new line.
        :param trial_folder: Absolute path to folder in which to place patient folds. Corresponds to a particular trial.
        '''
        sliding_df = self.clip_df[self.clip_df['pleural_line_findings'] != 'absent_lung_sliding']
        no_sliding_df = self.clip_df[self.clip_df['pleural_line_findings'] == 'absent_lung_sliding']

        # Get unique patient ids.
        sliding_ids = sliding_df['patient_id'].unique()
        no_sliding_ids = no_sliding_df['patient_id'].unique()

        print("{} unique sliding patient ids found.".format(len(sliding_ids)))
        print("{} unique no-sliding patient ids found.\n".format(len(no_sliding_ids)))

        # Shuffle the patient ids.
        np.random.shuffle(sliding_ids)
        np.random.shuffle(no_sliding_ids)

        trial_name = trial_folder.split('\\')[-1]
        folds_path = os.path.join(trial_folder, add_date_to_filename('folds_' + trial_name) + '.txt')

        # Make the file that stores ids by fold if it doesn't already exist.
        # Clear the file if user intends to overwrite its contents.
        overwrite = cfg['PARAMS']['OVERWRITE_FOLDS']
        if not os.path.exists(folds_path):
            open(folds_path, "x").close()
        elif overwrite:
            open(folds_path, "w").close()

        # Populate the file to store ids by fold.
        folds = []
        cur_fold = []

        # Keep track of the desired sizes for each fold. Since not all folds will be of the same size in terms
        # of patient ids, this eliminates the need to dynamically compute each fold size.
        n_folds = cfg['FOLD_SAMPLE']['NUM_FOLDS']
        fold_sizes = [len(sliding_ids) // n_folds] * n_folds
        fold_sizes[-1] += len(sliding_ids) % n_folds

        # Note: the code block above assumes that, if the number of sliding ids is not perfectly divisible by the
        # desired number of folds, then we make one fold slightly bigger than the rest. This must be modified if
        # we choose not to keep the smaller "leftover" fold.

        sliding_count = 0
        sliding = []
        no_sliding = []

        print("Sampling folds...\n")
        # Populate each fold with sliding ids
        cur_fold_num = 0
        for size in fold_sizes:
            for i in range(size):
                cur_id = sliding_ids[-1]
                sliding_ids = sliding_ids[:-1]
                clips_by_id = sliding_df[sliding_df['patient_id'] == cur_id]['id'].values
                sliding_count += len(clips_by_id)
                cur_fold.extend(clips_by_id)
            folds.append(cur_fold)
            cur_fold = []
            cur_fold_num += 1
            sliding.append(sliding_count)
            sliding_count = 0

        num_folds = len(folds)

        # Distribute non-sliding ids across folds. This is done in a roulette-like manner so that the ratio of negative
        # to positive classes in each fold is preserved.
        i = 0
        while True:
            if len(no_sliding_ids) == 0:
                break
            cur_id = no_sliding_ids[-1]
            no_sliding_ids = no_sliding_ids[:-1]
            clips_by_id = no_sliding_df[no_sliding_df['patient_id'] == cur_id]['id'].values
            no_sliding_count = len(clips_by_id)
            folds[i % num_folds].extend(clips_by_id)
            if i < num_folds:
                no_sliding.append(no_sliding_count)
            else:
                no_sliding[i % num_folds] += no_sliding_count
            i += 1
        for i in range(num_folds):
            print("Fold {}: {} clips with lung sliding, {} clips without lung sliding. Negative:Positive ~ {}"
                  .format(i + 1, sliding[i], no_sliding[i], round(sliding[i] / no_sliding[i], 3)))

        avg_neg_to_pos_ratio = sum([sliding[i] / no_sliding[i] for i in range(len(folds))]) / len(folds)
        print("Average negative:positive ratio ~ {}\n".format(round(avg_neg_to_pos_ratio)))

        # Write the folds to the fold file:
        if cfg['FOLD_SAMPLE']['OVERWRITE_FOLDER']:
            refresh_folder(os.getcwd() + cfg['PATHS']['FOLDS'])
        write_folds_to_txt(folds, folds_path)

        print("Sampling complete with seed = " + str(cfg['FOLD_SAMPLE']['SEED']) +
              ". Folds saved to " + folds_path + ".")

    def finetune_single_trial(self, trial_folder, hparams=None, lazy=True):
        '''
        Performs a single fine-tuning trial. Fine-tunes a model on external clip data, repeatedly augmenting external
        data slices to a training set until all external data has been used for training.
        :param trial_folder: Absolute path to folder in which to place metrics for current trial.
        :param hparams: Specifies the set of hyperparameters for fine-tuning the model.
        :param lazy: When True, trial will terminate once model performance exceeds minimum thresholds.
        '''

        # Set up result directories for given trial; read values from config file.
        model_out_dir = os.path.join(trial_folder, 'models')
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)

        model_name = cfg_full['TRAIN']['MODEL_DEF'].lower()
        hparams = cfg_full['TRAIN']['PARAMS'][model_name.upper()] if hparams is None else hparams
        ext_folder = os.getcwd() + cfg['PATHS']['CSVS_SPLIT']

        # Retrieve patient folds (stored by clip ids).
        filename = glob.glob(os.path.join(trial_folder, '*.txt'))[0]
        folds_file = os.path.join(trial_folder, filename)
        folds = open(folds_file, "r")
        cur_fold = []
        # stores the folds extracted from the txt file.
        fold_list = []

        # Extract the folds from the folds txt
        while True:
            fold_len = folds.readline()
            # Reached end of file.
            if fold_len == '':
                break
            fold_len = int(fold_len)
            for i in range(fold_len):
                cur_fold.append(folds.readline()[:-1])
            fold_list.append(cur_fold)
            cur_fold = []

        print("Retrieved {} external patient folds. The lengths of each fold are: ".format(len(fold_list)),
              [len(fold) for fold in fold_list])

        # set up the labelled external dataframe for augmenting the training data
        ext_dfs = []
        for center in cfg['LOCATIONS']:
            center_split_path = os.path.join(ext_folder, center)
            for csv in ['train.csv', 'val.csv']:
                ext_dfs.append(pd.read_csv(os.path.join(center_split_path, csv)))
        ext_df = pd.concat(ext_dfs, ignore_index=True)
        labelled_ext_df_path = os.path.join(os.getcwd() + cfg['PATHS']['CSV_OUT'], 'labelled_external_dfs/')
        if cfg['REFRESH_FOLDERS']:
            refresh_folder(labelled_ext_df_path)
        labelled_ext_df_path += add_date_to_filename('labelled_ext') + '.csv'
        ext_df.to_csv(labelled_ext_df_path)

        train_df = pd.DataFrame([])

        print('Setup complete')

        # Fine-tuning trial starts here.
        cur_fold_num = 0
        num_folds_added = 0  # number of external data slices that have been added to the training set
        min_test_size = int(len(ext_df) * cfg['MIN_TEST_SIZE'])

        # After .npz conversion, clip ids are followed by indices. Ignore these indices for patient id retrieval by
        # clip id.
        original_clip_ids = pd.DataFrame(ext_df['id'].str[:-2], columns=['id'])
        clip_id_count = original_clip_ids.groupby('id').transform('count')
        original_clip_ids = pd.concat([original_clip_ids, clip_id_count], axis=1)
        ext_df = ext_df.drop(columns=['id'])
        ext_df = pd.concat([original_clip_ids, ext_df], axis=1)
        
        trial_results_df = pd.DataFrame([])

        # This will later become a column for external data proportions in the trial result df.
        props = [0]

        # Loop until external testing set has reached minimum size.
        while len(ext_df) >= min_test_size:
            print('\n======================= FINE-TUNING ON {} OF {} SLICES OF EXTERNAL DATA ========================\n'
                  .format(str(num_folds_added + 1), len(fold_list)))
            # Evaluate the model on the external test set. Note that we load the original pre-finetuned model at each
            # iteration to minimize co-dependence between runs of a given trial.
            print('Evaluating model performance before fine-tuning...')
            # Can change these metrics as necessary
            metrics = [Recall(name='sensitivity'), Specificity(name='specificity')]

            # Get evaluation results on model before fine-tuning.
            original_model_path = os.path.join(os.getcwd(), cfg_full['PREDICT']['MODEL'])
            res_df = get_eval_results(original_model_path, ext_df, metrics)

            model = load_model(original_model_path, compile=False)

            # Read original model params from config file. Store into model_config.
            model_config = cfg_full['TRAIN']['PARAMS']['EFFICIENTNET']

            # Save the model's loss/sens/spec results from before fine-tuning.
            if num_folds_added == 0:
                trial_results_df = pd.concat([trial_results_df, res_df])
                # Save the original model in the trial's model directory
                if not lazy or check_performance(res_df):
                    print('Saving original model...')
                    model.save(os.path.join(model_out_dir, 'model_0.0_percent_train'))
                    if lazy:
                        return

            # Set aside some external data to be added to the existing training data.
            add_set = fold_list[num_folds_added]
            add_df = ext_df[ext_df['id'].isin(add_set)]

            # Remove the augmentation data from the external dataset. Becomes the new testing set.
            ext_df = ext_df[~ext_df['id'].isin(add_df['id'])]

            # Augment the training data.
            train_df = pd.concat([train_df, add_df])

            # Shuffle the training data.
            train_df.sample(frac=1)

            # Print the ratio of external train vs external test data
            print('Train: {}\nTest: {}'.format(len(train_df), len(ext_df)))

            # Create training and validation sets for fine-tuning.
            sub_val_df = train_df.tail(int(len(train_df) * cfg['FINETUNE']['VAL_SPLIT']))
            sub_train_df = train_df[~train_df.isin(sub_val_df['id'])]

            # Model fine-tuning.
            print('Fine-tuning...')

            # The following code was borrowed from src/models/models.py.
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
            tensorboard_path = cfg_full['TRAIN']['PATHS']['TENSORBOARD']
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path)

            # Log metrics
            log_dir = add_date_to_filename(os.getcwd() + cfg_full['TRAIN']['PATHS']['TENSORBOARD'])

            # uncomment line below to include tensorboard profiler in callbacks
            # basic_call = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=1)
            basic_call = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Learning rate scheduler & logging LR
            writer1 = tf.summary.create_file_writer(log_dir + '/train')

            scheduler = make_scheduler(writer1, hparams)
            lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

            # Creating a ModelCheckpoint for saving the model
            save_cp = ModelCheckpoint(os.path.join(model_out_dir, 'model_'
                                                   + str((num_folds_added + 1) / cfg['FOLD_SAMPLE']['NUM_FOLDS'])
                                                   + '_percent_train'),
                                      save_best_only=cfg_full['TRAIN']['SAVE_BEST_ONLY'])

            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg_full['TRAIN']['PATIENCE'],
                                           mode='min',
                                           restore_best_weights=True, min_delta=cfg['FINETUNE']['MIN_DELTA'])

            # Log model params to tensorboard
            writer2 = tf.summary.create_file_writer(log_dir + '/test')
            if log_dir is not None:
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
            current_results = get_eval_results(current_model_path, ext_df, metrics)
            trial_results_df = pd.concat([trial_results_df, current_results])
            props.append((num_folds_added + 1) / cfg['FOLD_SAMPLE']['NUM_FOLDS'])

            # If the fine-tuned model now performs well on the test set, terminate the trial early.
            perf_check = check_performance(current_results)
            if perf_check and lazy:
                break

            # # Remove model if performance not satisfactory.
            # elif not perf_check:
            #     rmtree(current_model_path)

            gc.collect()
            tf.keras.backend.clear_session()
            del model

            if len(ext_df) < min_test_size:
                print('EVALUATING PREVIOUS MODELS ON MINIMUM SIZED TEST SET...')

                # Set up column in trial results df to specify whether results are for fixed or variable sized test set.
                var_size_test_set = [1] * (num_folds_added + 1)
                # Add an extra 1 in the test set size column. This is for when we evaluated the original (non finetuned)
                # model on the external dataset.
                var_size_test_set.append(1)

                # For each previous model, get results on minimum (fixed) size test set.
                for prev_model in os.listdir(model_out_dir):
                    prev_model_path = os.path.join(model_out_dir, prev_model)
                    fixed_test_set_eval = get_eval_results(prev_model_path, ext_df, metrics)
                    trial_results_df = pd.concat([trial_results_df, fixed_test_set_eval])
                    var_size_test_set.append(0)

                # Concatenate fixed/variable size test set column with the trial result df.
                var_size_test_set = pd.DataFrame(var_size_test_set, columns=['variable_sized_test_set'])\
                    .reset_index(drop=True)
                props.extend(props)
                props = pd.DataFrame(props, columns=['ext_data_prop']).reset_index(drop=True)
                trial_results_df = trial_results_df.reset_index(drop=True)
                trial_results_df = pd.concat([var_size_test_set, props, trial_results_df], axis=1)
                trial_results_df.to_csv(os.path.join(trial_folder, 'sens_spec_results.csv'))

            # Update loop counters.
            cur_fold_num += 1
            num_folds_added += 1

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
        finetune_results_to_csv(experiment_path)


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

