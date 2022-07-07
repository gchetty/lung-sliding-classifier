import os
import sys
import yaml
import gc
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from src.preprocessor import MModePreprocessor
from models.models import get_model
from train import log_train_params
from predict import predict_set
from src.custom.metrics import Specificity, PhiCoefficient
from src.data.utils import refresh_folder
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, \
    FalsePositives, Accuracy, SensitivityAtSpecificity
from tensorflow_addons.metrics.f_scores import F1Score
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

# cfg refers to the subdictionary of the config file pertaining to our fine-tuning experiments. cfg_full is the entire
# loaded config file.
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))['GENERALIZE']
cfg_full = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))


def make_scheduler(writer, hparams):
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
    Saves patient id folds to .txt file in the following structure:
    -> An integer, n, which indicates the length of the current fold
    -> Followed by n lines, each containing one (unique) patient id
    '''
    txt = open(file_path, "a")
    for i in range(len(folds)):
        txt.write(str(len(folds[i])) + '\n')
        for id in folds[i]:
            txt.write(str(id) + '\n')
    txt.close()


def add_date_to_filename(file):
    '''
    Labels a file name with useful information about date and time.
    :param file: name of file as string
    :return labelled_filename: name of file joined with date and time info.
    '''
    cur_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    labelled_filename = file + '_' + cur_date
    return labelled_filename


def check_performance(df):
    '''
    Returns true if all metrics stored in the supplied results DataFrame meet fine-tuning standards, or False otherwise.
    :param df: DataFrame storing model performance metrics to be evaluated against fine-tune performance thresholds.
    '''
    # Return false if values less than these thresholds
    lower_bounds = ['accuracy', 'auc', 'recall', 'precision', 'true_negatives', 'true_positives']
    # Return false if values greater than these thresholds
    upper_bounds = ['false_negatives', 'true_positives']

    for thresh in lower_bounds:
        if df[thresh].values[0] < cfg['FINETUNE']['METRIC_THRESHOLDS'][thresh.upper()]:
            return False
    for thresh in upper_bounds:
        if df[thresh].values[0] > cfg['FINETUNE']['METRIC_THRESHOLDS'][thresh.upper()]:
            return False

    return True


# TAAFT stands for: Threshold-Aware Accumulative Fine-Tuner
class TAAFT:
    '''
    The CucumberTrainer class encapsulates the implementation of the fine-tuning for generalizability experiments.
    Handles fold sampling and fine-tuning trials.
    '''
    def __init__(self, clip_df, k):
        '''
        :param df: DataFrame containing all clip data from external centers
        :param k: Number of folds in which to split data in df by patient id
        '''
        print("CREATING A NEW CUCUMBER TRAINER (k = {})".format(k))
        self.k = k
        self.clip_df = clip_df

    def sample_folds(self):
        '''
        Samples external data in self.clip_df into self.k folds. Preserves ratio of positive to negative classes in each fold.
        Folds are saved in a .txt file, where each fold is identified with an integer index, followed by each patient id
        in the fold on a new line.
        '''
        sliding_df = self.clip_df[self.clip_df['pleural_line_findings'] != 'absent_lung_sliding']
        no_sliding_df = self.clip_df[self.clip_df['pleural_line_findings'] == 'absent_lung_sliding']

        # Get unique patient ids.
        sliding_ids = sliding_df.patient_id.unique()
        no_sliding_ids = no_sliding_df.patient_id.unique()

        print("{} unique sliding patient ids found.".format(len(sliding_ids)))
        print("{} unique no-sliding patient ids found.\n".format(len(no_sliding_ids)))

        # Shuffle the patient ids.
        np.random.seed(cfg_full['GENERALIZE']['FOLD_SAMPLE']['SEED'])
        np.random.shuffle(sliding_ids)
        np.random.shuffle(no_sliding_ids)

        # Specify the path to this file based on the config file.
        folds_dir = cfg['PATHS']['FOLDS']
        if not os.path.exists(folds_dir):
            os.makedirs(folds_dir)
        filename = ('patient_folds_' + str(datetime.datetime.now()) + '.txt').replace(' ', '_')
        str_parts = filename.split('.')
        filename = '.'.join([str_parts[0], str_parts[2]])
        filename = filename.replace(':', '-')
        folds_path = os.path.join(cfg['PATHS']['FOLDS'], filename)

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
                cur_fold.append(cur_id)
                clips_by_id = sliding_df[sliding_df['patient_id'] == cur_id]
                sliding_count += len(clips_by_id)
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
            folds[i % num_folds].append(cur_id)
            no_sliding_ids = no_sliding_ids[:-1]
            clips_by_id = no_sliding_df[no_sliding_df['patient_id'] == cur_id]
            no_sliding_count = len(clips_by_id)
            if i < num_folds:
                no_sliding.append(no_sliding_count)
            else:
                no_sliding[i % num_folds] += no_sliding_count
            i += 1
        for i in range(num_folds):
            print("Fold {}: There are {} clips with lung sliding, and {} clips without lung sliding. Negative:Positive ~ {}"
                  .format(i + 1, sliding[i], no_sliding[i], round(sliding[i] / no_sliding[i], 3)))

        avg_neg_to_pos_ratio = sum([sliding[i] / no_sliding[i] for i in range(len(folds))]) / len(folds)
        print("Average negative:positive ratio ~ {}\n".format(round(avg_neg_to_pos_ratio)))

        # Write the folds to the fold file:
        if cfg['FOLD_SAMPLE']['OVERWRITE_FOLDER']:
            refresh_folder(cfg['PATHS']['FOLDS'])
        write_folds_to_txt(folds, folds_path)

        print("Sampling complete with seed = " + str(cfg['FOLD_SAMPLE']['SEED']) + ". Folds saved to " + folds_path + ".")

    def finetune_single_trial(self, hparams=None):
        '''
        Performs a single fine-tuning trial. Fine-tunes a model on external clip data, repeatedly augmenting external
        data slices to a training set until all external data has been used for training.
        :param hparams: Specifies the set of hyperparameters for fine-tuning the model.
        '''

        print('Reading values from the config file...')

        n_folds = cfg['FOLD_SAMPLE']['NUM_FOLDS']

        model_out_dir = cfg['PATHS']['MODEL_OUT']
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)

        if cfg['FINETUNE']['OVERWRITE_MODEL_DIR']:
            refresh_folder(model_out_dir)

        metrics = cfg_full['CROSS_VAL']['METRICS']
        metrics_df = pd.DataFrame(np.zeros((n_folds + 2, len(metrics) + 1)), columns=['Fold'] + metrics)
        metrics_df['Fold'] = list(range(n_folds)) + ['mean', 'std']

        model_name = cfg_full['TRAIN']['MODEL_DEF'].lower()
        hparams = cfg_full['TRAIN']['PARAMS'][model_name.upper()] if hparams is None else hparams

        CSVS_FOLDER = cfg_full['TRAIN']['PATHS']['CSVS']
        EXT_FOLDER = cfg['PATHS']['CSVS_SPLIT']

        FOLDS_PATH = cfg['PATHS']['FOLDS']

        # Ensure that the partition path has been created before running this code!
        folds_file = os.path.join(FOLDS_PATH, os.listdir(FOLDS_PATH)[0])
        print('Retrieving patient folds from {}...'.format(folds_file))
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
        print('Reading external clip filenames and labels into dataframe...')

        # set up the labelled external dataframe for augmenting the training data
        ext_dfs = []
        for center in cfg['LOCATIONS']:
            center_split_path = os.path.join(EXT_FOLDER, center)
            for csv in ['train.csv', 'val.csv']:
                ext_dfs.append(pd.read_csv(os.path.join(center_split_path, csv)))
        ext_df = pd.concat(ext_dfs, ignore_index=True)
        labelled_ext_df_path = os.path.join(cfg['CSV_OUT'], 'labelled_external_dfs/')
        if cfg['REFRESH_FOLDERS']:
            refresh_folder(labelled_ext_df_path)
        labelled_ext_df_path += add_date_to_filename('labelled_ext')
        ext_df.to_csv(labelled_ext_df_path)

        train_df = pd.DataFrame([])

        print('Setup complete')

        # Cucumber trial starts here.
        cur_fold_num = 0
        num_folds_added = 0  # number of external data slices that have been added to the training set
        add_set = []
        # Loop until all external data has been used up.
        while num_folds_added < cfg['FOLD_SAMPLE']['NUM_FOLDS']:
            print('\n========================== STARTING TRIAL {} OF MAXIMUM {} =============================\n'
                  .format(str(num_folds_added + 1), len(fold_list)))
            # Evaluate the model on the external test set. Note that we load the original pre-finetuned model at each
            # iteration to minimize codependence between runs of a given trial.
            model = load_model(cfg_full['PREDICT']['MODEL'], compile=False)
            pred_labels, probs = predict_set(model, ext_df)
            # Can change these metrics as necessary
            metrics = [Accuracy(name='accuracy'), AUC(), F1Score(num_classes=2, average='micro', threshold=0.5), Precision(name='precision'), Recall(name='recall'),
                       TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives(), Specificity(name='specificity'),
                       PhiCoefficient(), SensitivityAtSpecificity(0.95)]

            # Create a DataFrame containing metric results from predictions.
            results = []
            metric_names = []
            for metric in metrics:
                # Probabilities are stored as floats, while Accuracy metric compares integer values for labels and
                # probs. Round probabilities to the closer value of 0 or 1.
                if metric.name == 'Accuracy':
                    probs = np.rint(probs)
                metric.update_state(ext_df['label'], probs)
                results += [metric.result().numpy()]
                metric_names += [metric.name]
            res_df = pd.DataFrame([results], columns=metric_names)

            predictions_path = cfg['PATHS']['PREDICTIONS_OUT']
            if not os.path.exists(predictions_path):
                os.makedirs(predictions_path)

            if cfg['REFRESH_FOLDERS']:
                refresh_folder(predictions_path)

            # Save the metric result DataFrame.
            csv_name = add_date_to_filename('') + '_metrics.csv'
            res_df.to_csv(os.path.join(predictions_path, csv_name), index=False)
            print("Metric results saved to", predictions_path)

            # If the model performs well on the external test set, save it! (Stop the trial here.)
            if check_performance(res_df):
                print('Model performance ok. Saving model...')
                model.save(os.path.join(model_out_dir, add_date_to_filename('finetuned_model') + '.pb'))
                write_folds_to_txt(add_set, cfg['PATHS']['FOLDS_USED'])
                continue

            # Set aside some external data to be added to the existing training data.
            add_set = fold_list[num_folds_added]
            print(len(add_set))
            add_df = ext_df[ext_df['patient_id'].isin(add_set)]

            # Remove the augmentation data from the external dataset. Becomes the new testing set.
            ext_df = ext_df[~ext_df['patient_id'].isin(add_set)]
            print('\n{} records removed from the external dataset for training.'.format(len(add_df)))

            # Augment the training data.
            train_df = pd.concat([train_df, add_df])

            # Shuffle the training data.
            train_df.sample(frac=1)
            print('Training clips: {}\nValidation clips: {}'.format(len(train_df), len(ext_df)))

            # Create training and validation sets for fine-tune cross-validation.
            k = self.k
            val_num = len(train_df) // k
            train_folds = []
            i = 0
            while i < len(train_df):
                train_folds.append(train_df[i:i + min(len(train_df) - i, val_num)])
                i += val_num
            print("{} folds ready for cross-val.".format(len(train_folds)))

            # Cross-validation for model fine-tuning.
            print('Fine-tune cross-validation commencing...')
            for cv_run in range(k):
                # pick up training from where it left off at the end of previous cross val run.
                if cv_run > 0:
                    prev_model_path = os.path.join(cfg['PATHS']['MODEL_OUT'], os.listdir(cfg['PATHS']['MODEL_OUT'])[-1])
                # for run 0 of cross val, retrieve the model from its location immediately after initial training.
                else:
                    prev_model_path = os.path.join(cfg_full['TRAIN']['PATHS']['MODEL_OUT'],
                                                   os.listdir(cfg_full['TRAIN']['PATHS']['MODEL_OUT'])[-1])
                model = load_model(prev_model_path, compile=False)

                print('FINE-TUNING MODEL FOR FOLD ' + str(cv_run) + ' OF EXTERNAL CLIP TRAINING SET')

                # Set up the train and holdout for each cross-validation run. Note that both parts are taken from the
                # external data set aside for training. We have a separate dataframe to store testing data.
                df_val_part = train_folds[cv_run]
                df_train_part = train_df[~train_df.isin(df_val_part['patient_id'].unique())]

                # The following code was borrowed from src/models/models.py.
                # Fine-tune the model with a cross-validation run
                model_def_fn, preprocessing_fn = get_model(cfg['FINETUNE']['MODEL_DEF'])

                # Prepare the training and validation sets from the dataframes.
                train_set = tf.data.Dataset.from_tensor_slices((df_train_part['filename'].tolist(), df_train_part['label'].tolist()))
                val_set = tf.data.Dataset.from_tensor_slices((df_val_part['filename'].tolist(), df_val_part['label'].tolist()))

                preprocessor = MModePreprocessor(preprocessing_fn)

                # Define the preprocessing pipelines for train, test and validation
                train_set = preprocessor.prepare(train_set, df_train_part, shuffle=True, augment=True)
                val_set = preprocessor.prepare(val_set, df_val_part, shuffle=False, augment=False)

                # Create dictionary for class weights like in src/train.py.
                num_no_sliding = len(train_df[train_df['label'] == 0])
                num_sliding = len(train_df[train_df['label'] == 1])
                total = num_no_sliding + num_sliding
                weight_for_0 = (1 / num_no_sliding) * (total / 2.0)
                weight_for_1 = (1 / num_sliding) * (total / 2.0)
                class_weight = {0: weight_for_0, 1: weight_for_1}

                # Defining Binary Classification Metrics

                # Refresh the TensorBoard directory
                tensorboard_path = cfg_full['TRAIN']['PATHS']['TENSORBOARD']
                if not os.path.exists(tensorboard_path):
                    os.makedirs(tensorboard_path)

                # Log metrics
                log_dir = add_date_to_filename(cfg_full['TRAIN']['PATHS']['TENSORBOARD'])

                # uncomment line below to include tensorboard profiler in callbacks
                # basic_call = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=1)
                basic_call = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                # Learning rate scheduler & logging LR
                writer1 = tf.summary.create_file_writer(log_dir + '/train')

                scheduler = make_scheduler(writer1, hparams)
                lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

                # Creating a ModelCheckpoint for saving the model
                save_cp = ModelCheckpoint(os.path.join(model_out_dir, add_date_to_filename('models')), save_best_only=cfg_full['TRAIN']['SAVE_BEST_ONLY'])

                # Early stopping
                early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg_full['TRAIN']['PATIENCE'],
                                               mode='min',
                                               restore_best_weights=True)

                # Log model params to tensorboard
                writer2 = tf.summary.create_file_writer(log_dir + '/test')
                if log_dir is not None:
                    log_train_params(writer1, hparams)

                model_config = cfg_full['TRAIN']['PARAMS']['EFFICIENTNET']
                lr = model_config['LR']

                optimizer = Adam(learning_rate=lr)
                model.compile(loss=SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO,
                                                                       alpha=model_config['ALPHA'],
                                                                       gamma=model_config['GAMMA']),
                              optimizer=optimizer, metrics=metrics)

                # Train and save the model
                epochs = cfg['FINETUNE']['CROSS_VAL']['EPOCHS']
                history = model.fit(train_set, epochs=epochs, validation_data=val_set,
                                            class_weight=class_weight,
                                            callbacks=[save_cp, basic_call, early_stopping, lr_callback], verbose=1)

                # Log early stopping to tensorboard
                if history is not None:
                    if len(history.epoch) < epochs:
                        with writer2.as_default():
                            tf.summary.text(name='Early Stopping', data=tf.convert_to_tensor('Training stopped early'),
                                            step=0)

                for metric in history.history.keys():
                    if metric in metrics_df.columns:
                        metric_values = history.history[metric]
                        metrics_df.loc[cv_run, metric] = sum(metric_values) / len(metric_values)
                gc.collect()
                tf.keras.backend.clear_session()
                del model

            # Save results
            file_path = os.path.join(cfg['PATHS']['EXPERIMENTS'], add_date_to_filename('cross_val') + '.csv')
            metrics_df.to_csv(file_path, columns=metrics_df.columns, index_label=False, index=False)

            cur_fold_num += 1
            num_folds_added += 1

    def finetune_multiple_trials(self, n_trials):
        '''
        Runs multiple trials.
        :param n_trials: Number of trials to run.
        '''
        pass


if __name__ == '__main__':
    # If the generalizability folder has not been created, do this first. Create the subdirectories for each of the
    # required data types (e.g. raw clips, npzs, m-modes, etc).
    generalize_path = '/'.join(cfg['PATHS']['CSV_OUT'].split('/')[:-1])
    if not os.path.exists(generalize_path):
        os.makedirs(generalize_path)
    for path in cfg['PATHS']:
        if not os.path.exists(cfg['PATHS'][path]):
            os.makedirs(cfg['PATHS'][path])

    # Insert code to automatically preprocess everything?
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
    os.system('python')

    # Consolidate all external clip data into a single dataframe.
    csvs_dir = cfg['PATHS']['CSV_OUT']
    external_data = []
    for center in cfg['LOCATIONS']:
        csv_folder = os.path.join(csvs_dir, center)
        for input_class in ['sliding', 'no_sliding']:
            csv_file = os.path.join(csv_folder, input_class + '.csv')
            external_data.append(pd.read_csv(csv_file))
    external_df = pd.concat(external_data)

    external_data_dir = os.path.join(cfg['PATHS']['CSV_OUT'], 'combined_external_data')
    external_df_path = os.path.join(external_data_dir, add_date_to_filename('all_external_clips') + '.csv')

    if cfg['REFRESH_FOLDERS']:
        refresh_folder(external_data_dir)

    external_df.to_csv(external_df_path)
    print("All external clips consolidated into", external_df_path, '\n')
    # Construct a CucumberTrainer instance with the dataframe.
    taaft = TAAFT(external_df, 5)
    taaft.sample_folds()
    taaft.finetune_single_trial()
