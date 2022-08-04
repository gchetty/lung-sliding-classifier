import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import glob
import numpy as np
from shutil import rmtree

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))['EXTERNAL_VAL']


def finetune_results_to_csv(experiment_path):
    '''
    Saves finetune results for each trial (and for each fold in that trial) as a csv file.
    :param experiment_path: Absolute path to fine-tuning experiment files.
    :return Str: Absolute path to the .csv file containing finetuning results across all trials.
    '''
    # trial_results is going to be the combined results .csv file.
    trial_results = pd.DataFrame([])
    trial_folders = glob.glob(os.path.join(experiment_path, 'trial_[0-9]*'))

    # trial_numbers specifies which trial each result row is for in the .csv file.
    trial_numbers = []
    cur_trial_num = 1

    # Combine results across all trials.
    for trial in trial_folders:
        metric_results_csv_path = glob.glob(os.path.join(trial, '*.csv'))[0]
        metric_results_csv = pd.read_csv(metric_results_csv_path, index_col=0)
        trial_numbers.extend([cur_trial_num] * len(metric_results_csv))
        trial_results = pd.concat([trial_results, metric_results_csv])
        cur_trial_num += 1

    # Add trial_numbers as a column to the combined results df. Save as .csv file.
    trial_numbers = pd.DataFrame(trial_numbers, columns=['trial']).reset_index(drop=True)
    trial_results = trial_results.reset_index(drop=True)
    trial_results = pd.concat([trial_numbers, trial_results], axis=1)
    experiment_name = experiment_path.split('\\')[-1]
    final_path = os.path.join(experiment_path, experiment_name + '_results.csv')
    trial_results.to_csv(final_path)

    return final_path


def plot_finetune_results(results_csv):
    '''
    Plots fine-tuning results for metrics specified in the config file (currently loss/sensitivity/specificity).
    Generates a plot for each metric in results_csv by trial, along with trial-wise averages for each metric.
    :param results_csv: Absolute path to .csv file containing fine-tuning results.
    '''
    results_df = pd.read_csv(results_csv)

    # The absolute path to the corresponding experiment is the parent directory for the results .csv file.
    experiment_path = '\\'.join(results_csv.split('\\')[:-1])

    # Make a folder to store the plots.
    plot_dir = os.path.join(experiment_path, 'plots')

    # If there are already plots in the experiment folder, overwrite with new plots.
    if os.path.exists(plot_dir):
        rmtree(plot_dir)

    os.makedirs(plot_dir)

    # Get list of external data proportions used for training in the experiment.
    first_trial_results = results_df[results_df['trial'] == 1]
    ext_data_props = first_trial_results[first_trial_results['variable_sized_test_set'] == 1]['ext_data_prop'].values

    metrics = cfg['PLOT_METRICS']
    num_trials = cfg['NUM_TRIALS']
    plot_ind = 0
    legend_labels = []

    for metric in metrics:
        # New figure.
        plt.figure(plot_ind)
        plt.xlabel('Proportion of External Data Used for Train')
        metric_name = metric[0].upper() + metric[1:]
        plt.ylabel(metric_name)
        plt.title(metric_name + ' vs External Train Proportion')

        for test_set_type in [1, 0]:
            style = 'solid'
            if test_set_type == 0:
                style = 'dashed'
            results_for_test_set_type = results_df[results_df['variable_sized_test_set'] == test_set_type]
            test_set_size_label = '({} Test Set Size)'.format('Variable' if test_set_type == 1 else 'Fixed')

            # Plot results for each trial.
            for t in range(num_trials):
                results_to_plot = results_for_test_set_type[results_for_test_set_type['trial'] == t + 1]

                # Plot results for each of variable and fixed test set sizes.
                trial_metric_vals = results_to_plot[metric].values
                plt.plot(ext_data_props, trial_metric_vals, linestyle=style)

                legend_labels.append('Trial {} {}'.format(t + 1, test_set_size_label))

            # Plot average metric value across trials.
            trial_avgs = []
            for prop in ext_data_props:
                avg_for_prop = np.mean(results_for_test_set_type[results_for_test_set_type['variable_sized_test_set'] ==
                                                                 test_set_type][metric])
                trial_avgs.append(avg_for_prop)

            # Trial-wise averages are denoted with bold lines.
            plt.plot(ext_data_props, trial_avgs, linestyle=style, linewidth=3)

            legend_labels.append('Trial-Wise Average {}'.format(test_set_size_label))

        # Add a legend
        plt.legend(legend_labels, bbox_to_anchor=(1, 2))

        experiment_name = experiment_path.split('\\')[-1]
        plt.savefig(os.path.join(plot_dir, experiment_name + '_' + metric + '.png'))

        # Update plot figure counter.
        plot_ind += 1
        legend_labels = []


experiment_path = os.getcwd() + cfg['PATHS']['EXPERIMENTS']
experiment_path = os.path.join(experiment_path, os.listdir(experiment_path)[-1])
res_df = finetune_results_to_csv(experiment_path)

plot_finetune_results(res_df)