import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import glob
import numpy as np
from shutil import rmtree
import seaborn as sns

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))['EXTERNAL_VAL']


def finetune_results_to_csv(experiment_path, group=None):
    '''
    Saves finetune results for each trial (and for each fold in that trial) as a csv file.
    :param experiment_path: Absolute path to fine-tuning experiment files.
    :param group: If not None, specifies the subgroup whose results we'd like to summarize.
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
        if group is not None:
            metric_results_csv_path = glob.glob(os.path.join(trial, 'metrics_by_{}.csv'.format(group)))[0]
        else:
            metric_results_csv_path = glob.glob(os.path.join(trial, 'metrics.csv'))[0]
        metric_results_csv = pd.read_csv(metric_results_csv_path)
        trial_numbers.extend([cur_trial_num] * len(metric_results_csv))
        trial_results = pd.concat([trial_results, metric_results_csv])
        cur_trial_num += 1

    # Add trial_numbers as a column to the combined results df. Save as .csv file.
    trial_numbers = pd.DataFrame(trial_numbers, columns=['trial']).reset_index(drop=True)
    trial_results = trial_results.reset_index(drop=True)
    trial_results = pd.concat([trial_numbers, trial_results], axis=1)
    experiment_name = experiment_path.split('\\')[-1]
    if group is not None:
        final_path = os.path.join(experiment_path, experiment_name + '_results_by_{}.csv'.format(group))
    else:
        final_path = os.path.join(experiment_path, experiment_name + '_results.csv')
    trial_results.to_csv(final_path)

    return final_path


def plot_finetune_results(experiment_path, flip_labels=True, summarize=True, titles=None, plot_threshold=True):
    '''
    Plots fine-tuning results for metrics specified in the config file (currently loss/sensitivity/specificity).
    Generates a plot for each metric in results_csv by trial, along with trial-wise averages for each metric.
    :param experiment_path: Absolute path to the directory of the experiment whose results you'd like to plot
    :param flip_labels: Whether to flip the positive/negative class labels. If True, absent (present) lung sliding will
    be considered the positive (negative) class. If False, absent (present) lung sliding will be considered the negative
    (positive) class (as observed during training).
    :param summarize: If True, all metrics are plotted on one figure as individual subplots.
    :param titles: If not None, list of titles to give (sub)plot(s).
    :param plot_threshold: If True, the desired 'goal' metric is plotted as a horizontal line.
    '''
    results_csv = os.path.join(exp_path, os.path.basename(exp_path)+'_results.csv')
    if not os.path.exists(results_csv):
        results_csv = finetune_results_to_csv(experiment_path)

    results_df = pd.read_csv(results_csv)

    if flip_labels:
        results_df.rename(columns={'sensitivity':'specificity_temp','specificity':'sensitivity'},inplace=True)
        results_df.rename(columns={'specificity_temp':'specificity'}, inplace=True)

    # Make a folder to store the plots.
    plot_dir = os.path.join(experiment_path, 'plots')

    # If the plotting directory doesn't exist, make one
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Get list of external data proportions used for training in the experiment.
    first_trial_results = results_df[results_df['trial'] == 1]
    ext_data_props = first_trial_results[first_trial_results['variable_size_test_set'] == 1]['train_data_prop'].values

    metrics = cfg['PLOT_METRICS']
    num_trials = results_df['trial'].max()
    plot_ind = 0
    legend_labels = []

    fig_name = os.path.basename(experiment_path)

    if summarize:
        fig, axs = plt.subplots(1, len(metrics), figsize=(5 + 5*len(metrics),5))

    for metric in metrics:
        if not summarize:
            fig, ax = plt.subplots(1, 1, figsize=(10,5))
        else:
            ax = axs[plot_ind]

        if metric == 'sensitivity':
            thresh = 0.793
            ylimits = [0.3, 1.01]
        elif metric == 'specificity':
            thresh = 0.901
            ylimits = [0.3, 1.01]
        else:
            thresh = None
            ylimits = [0, 0.7]

        ax.set_xlabel('Proportion of External Data Used for Training', fontsize=12)
        metric_name = metric[0].upper() + metric[1:]
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xticks(ext_data_props, fontsize=12)
        ax.set_ylim(ylimits)
        if titles is not None:
            ax.set_title(titles[plot_ind])

        for test_set_type in [1, 0]:
            style = 'solid'
            marker = '*'
            if test_set_type == 0:
                style = 'dashed'
            results_for_test_set_type = results_df[results_df['variable_size_test_set'] == test_set_type]
            test_set_size_label = '({} Test Set)'.format('Variable' if test_set_type == 1 else 'Fixed')

            # Plot results for each trial.
            colours = plt.cm.plasma(np.linspace(0, 0.9, num_trials))
            for t in range(num_trials):
                results_to_plot = results_for_test_set_type[results_for_test_set_type['trial'] == t + 1]

                # Plot results for each of variable and fixed test set sizes.
                trial_metric_vals = results_to_plot[metric].values
                ax.plot(ext_data_props, trial_metric_vals, linestyle=style,color=colours[t],marker=marker,linewidth=1,markersize=6)

                legend_labels.append('Trial {} {}'.format(t + 1, test_set_size_label))

            # Plot average metric value across trials.
            trial_avgs = []
            trial_stds = []
            for prop in ext_data_props:
                avg_for_prop = np.mean(results_for_test_set_type[results_for_test_set_type['train_data_prop'] ==
                                                                 prop][metric])
                std_for_prop = np.std(results_for_test_set_type[results_for_test_set_type['train_data_prop'] ==
                                                                prop][metric])
                trial_avgs.append(avg_for_prop)
                trial_stds.append(std_for_prop)

            # Trial-wise averages are denoted in black
            ax.plot(ext_data_props, trial_avgs, linestyle=style, linewidth=2.5, color='k', zorder=100)

            legend_labels.append('Trial-Wise Mean {}'.format(test_set_size_label))

        # Add a legend
        if not summarize or plot_ind == len(metrics) - 1:
            ax.legend(legend_labels, fontsize=12, bbox_to_anchor=(1.05, 0.9))

        if plot_threshold and thresh is not None:
            ax.axhline(y=thresh)

        fig_name += '_' + metric

        if summarize:
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.2,
                                hspace=0.2)
            plt.tight_layout()
        else:
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, fig_name + '.png'))
            fig_name = os.path.basename(experiment_path)

        # Update plot figure counter.
        plot_ind += 1
        legend_labels = []

    if summarize:
        plt.savefig(os.path.join(plot_dir, fig_name + '.png'))

        return fig


def plot_subgroup_analysis_results(experiment_path, flip_labels=True, group='center', plot_threshold=True):
    '''
    Plots fine-tuning results for metrics specified in the config file (currently loss/sensitivity/specificity).
    Generates a plot for each metric in results_csv by trial, along with trial-wise averages for each metric.
    :param experiment_path: Absolute path to the directory of the experiment whose results you'd like to plot
    :param flip_labels: Whether to flip the positive/negative class labels. If True, absent (present) lung sliding will
    be considered the positive (negative) class. If False, absent (present) lung sliding will be considered the negative
    (positive) class (as observed during training).
    :param group (str): Specifies which subgroup we're interested in plotting
    :param plot_threshold: If True, the desired 'goal' metric is plotted as a horizontal line.
    '''
    results_csv = os.path.join(exp_path, os.path.basename(exp_path)+'_results_by_{}.csv'.format(group))
    if not os.path.exists(results_csv):
        results_csv = finetune_results_to_csv(experiment_path, group=group)

    results_df = pd.read_csv(results_csv)

    # If true, sets absent lung sliding as the positive class
    if flip_labels:
        results_df.rename(columns={'sensitivity': 'specificity_temp', 'specificity': 'sensitivity'}, inplace=True)
        results_df.rename(columns={'specificity_temp': 'specificity'}, inplace=True)

    # Designate plotting directory
    plot_dir = os.path.join(experiment_path, 'plots')

    # If the plotting directory doesn't exist, make one
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    metrics = cfg['PLOT_METRICS']
    num_trials = results_df['trial'].max()

    # Get list of external data proportions used for training in the experiment.
    first_trial_results = results_df[results_df['trial'] == 1]
    ext_data_props = first_trial_results[first_trial_results['variable_size_test_set'] == 1]['train_data_prop'].unique()

    fig_name = os.path.basename(experiment_path)

    fig, axs = plt.subplots(len(results_df[group].unique()),
                            len(metrics),
                            figsize=(5 + 5 * len(metrics),
                                     5 + 5*len(results_df[group].unique())))
    subgroup_ind = 0

    for subgroup in results_df[group].unique():
        df = results_df.loc[results_df[group] == subgroup]

        legend_labels = []
        metric_ind = 0
        for metric in metrics:
            ax = axs[subgroup_ind][metric_ind]

            if metric == 'sensitivity':
                thresh = 0.793
                ylimits = [0, 1.01]
            elif metric == 'specificity':
                thresh = 0.901
                ylimits = [0, 1.01]
            else:
                thresh = None
                ylimits = [0, 0.3]

            ax.set_xlabel('Proportion of External Data Used for Training', fontsize=16)
            metric_name = metric[0].upper() + metric[1:]
            ax.set_ylabel(metric_name, fontsize=16)
            ax.set_xticks(ext_data_props, fontsize=16)
            ax.set_ylim(ylimits)

            ax.set_title('{}: {}'.format(group,subgroup), fontsize=16)
            if plot_threshold and thresh is not None:
                ax.axhline(y=thresh, color='k')

            for test_set_type in [1, 0]:
                style = 'solid'
                marker = '*'
                if test_set_type == 0:
                    style = 'dashed'
                results_for_test_set_type = df[df['variable_size_test_set'] == test_set_type]
                test_set_size_label = '({} Test Set)'.format('Variable' if test_set_type == 1 else 'Fixed')

                # Plot results for each trial.
                colours = plt.cm.plasma(np.linspace(0, 0.9, num_trials))
                for t in range(num_trials):
                    results_to_plot = results_for_test_set_type[results_for_test_set_type['trial'] == t + 1]

                    # Plot results for each of variable and fixed test set sizes.
                    trial_metric_vals = results_to_plot[metric].values
                    ax.plot(ext_data_props, trial_metric_vals, linestyle=style, color=colours[t], marker=marker,
                            linewidth=1, markersize=6)

                    legend_labels.append('Trial {} {}'.format(t + 1, test_set_size_label))

                # Plot average metric value across trials.
                trial_avgs = []
                trial_stds = []
                for prop in ext_data_props:
                    avg_for_prop = np.mean(results_for_test_set_type[results_for_test_set_type['train_data_prop'] ==
                                                                     prop][metric])
                    std_for_prop = np.std(results_for_test_set_type[results_for_test_set_type['train_data_prop'] ==
                                                                    prop][metric])
                    trial_avgs.append(avg_for_prop)
                    trial_stds.append(std_for_prop)

                # Trial-wise averages are denoted in black
                ax.plot(ext_data_props, trial_avgs, linestyle=style, linewidth=2.5, color='k', zorder=100)

                legend_labels.append('Trial-Wise Mean {}'.format(test_set_size_label))

            # Update plot figure counter.
            metric_ind += 1
            legend_labels = []
        subgroup_ind += 1

    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, fig_name + '_results_by_{}.png'.format(group)))


def plot_mean_sensitivity_specificity_tradeoff(experiment_path, flip_labels=True,get_intersection_of=None,print_metrics=False):

    '''
    Plots fine-tuning results for metrics specified in the config file (currently loss/sensitivity/specificity).
    Generates a plot for each metric in results_csv by trial, along with trial-wise averages for each metric.
    :param experiment_path: Absolute path to the directory of the experiment whose results you'd like to plot
    :param flip_labels: Whether to flip the positive/negative class labels. If True, absent (present) lung sliding will
    be considered the positive (negative) class. If False, absent (present) lung sliding will be considered the negative
    (positive) class (as observed during training).
    :param get_intersection_of: If not None, specifies the test set ('fixed' or 'variable') on which to evaluate and
    plot intersection of specificity and sensitivity in terms of the external data proportion.
    :param print_metrics: If True, prints average metrics.
    '''

    results_csv = os.path.join(exp_path, os.path.basename(exp_path) + '_results.csv')
    if not os.path.exists(results_csv):
        results_csv = finetune_results_to_csv(experiment_path)

    results_df = pd.read_csv(results_csv)

    if flip_labels:
        results_df.rename(columns={'sensitivity':'specificity_temp','specificity':'sensitivity'},inplace=True)
        results_df.rename(columns={'specificity_temp':'specificity'}, inplace=True)

    # Make a folder to store the plots.
    plot_dir = os.path.join(experiment_path, 'plots')

    # If the plotting directory doesn't exist, make one
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Get list of external data proportions used for training in the experiment.
    first_trial_results = results_df[results_df['trial'] == 1]
    ext_data_props = first_trial_results[first_trial_results['variable_size_test_set'] == 1]['train_data_prop'].values

    metrics = ['sensitivity', 'specificity']
    legend_labels = []

    fig_name = os.path.basename(experiment_path) + '_compare_mean'

    fig, ax = plt.subplots(1, 1, figsize=(8,5))

    ax.set_xlabel('Proportion of External Data Used for Training', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_xticks(ext_data_props, fontsize=12)

    avg_specificity_variable = []
    avg_specificity_fixed = []
    avg_sensitivity_variable = []
    avg_sensitivity_fixed = []

    for metric in metrics:

        metric_name = metric[0].upper() + metric[1:]
        if metric_name == 'Specificity':
            colour = 'r'
        else:
            colour = 'tab:blue'

        for test_set_type in [1, 0]:
            style = 'solid'
            if test_set_type == 0:
                style = 'dashed'
            results_for_test_set_type = results_df[results_df['variable_size_test_set'] == test_set_type]
            test_set_size_label = '({} Test Set)'.format('Variable' if test_set_type == 1 else 'Fixed')

            # Plot average metric value across trials.
            trial_avgs = []
            trial_stds = []
            for prop in ext_data_props:
                avg_for_prop = np.mean(results_for_test_set_type[results_for_test_set_type['train_data_prop'] ==
                                                                 prop][metric])
                std_for_prop = np.std(results_for_test_set_type[results_for_test_set_type['train_data_prop'] ==
                                                                prop][metric])
                trial_avgs.append(avg_for_prop)
                trial_stds.append(std_for_prop)
            if test_set_type == 1:
                if metric_name == 'Specificity':
                    avg_specificity_variable = trial_avgs
                elif metric_name == 'Sensitivity':
                    avg_sensitivity_variable = trial_avgs
            else:
                if metric_name == 'Specificity':
                    avg_specificity_fixed = trial_avgs
                elif metric_name == 'Sensitivity':
                    avg_sensitivity_fixed = trial_avgs

            # Trial-wise averages are denoted in black
            ax.plot(ext_data_props, trial_avgs, linestyle=style, linewidth=2, color=colour, zorder=100)
            legend_labels.append('Mean Trial-Wise {} {}'.format(metric_name,test_set_size_label))

            if not (metric in fig_name):
                fig_name += '_' + metric

    if print_metrics:
        print('Mean sensitivity:')
        print('External data training proportion: {}'.format(list(ext_data_props)))
        print('Variable test set: {}'.format(avg_sensitivity_variable))
        print('Fixed test set: {}'.format(avg_sensitivity_fixed))
        print('\nMean specificity:')
        print('External data training proportion: {}'.format(list(ext_data_props)))
        print('Variable test set: {}'.format(avg_specificity_variable))
        print('Fixed test set: {}'.format(avg_specificity_fixed))

    # Add a legend
    ax.legend(legend_labels, fontsize=12, loc='best')

    if get_intersection_of is not None:

        if get_intersection_of == 'variable':
            x_itn, sens_itn, spec_itn = get_intersection_point(ext_data_props,
                                                               avg_sensitivity_variable,
                                                               avg_specificity_variable)
        elif get_intersection_of == 'fixed':
            x_itn, sens_itn, spec_itn = get_intersection_point(ext_data_props,
                                                               avg_sensitivity_fixed,
                                                               avg_specificity_fixed)

        ax.vlines(x_itn, 0.5, sens_itn, linestyle=":", linewidth=1.5, color='#424242')
        ax.vlines(x_itn, 0.5, spec_itn, linestyle=":", linewidth=1.5, color='#424242')
        ax.hlines(sens_itn, 0, x_itn, linestyle=":", linewidth=1.5, color='tab:blue')
        ax.plot(x_itn, sens_itn,linestyle=':',marker='o', color='tab:blue', markersize=9)
        ax.hlines(spec_itn, 0, x_itn, linestyle=":", linewidth=1.5, color='tab:red')
        ax.plot(x_itn, spec_itn, 'ro', markersize=9,zorder=200)
        ax.text(x_itn, sens_itn +0.025, "Training proportion = {}".format(x_itn),
            ha='right', va='center', fontsize=10, color='#424242')
        ax.text(x_itn, sens_itn +0.05, "Sensitivity = {:.3f}".format(sens_itn),
            ha='right', va='center', fontsize=10, color="tab:blue")
        ax.text(x_itn, sens_itn +0.075, "Specificity = {:.3f}".format(spec_itn),
            ha='right', va='center', fontsize=10, color="tab:red")

        plt.savefig(os.path.join(plot_dir, fig_name + '_' + get_intersection_of + '.png'))
    else:
        plt.savefig(os.path.join(plot_dir, fig_name + '_.png'))

    return fig


def get_intersection_point(x, y_sens, y_spec):
    '''
    Computes the intersection of sensitivity and specificity vectors.
    :param x: vector of external data proportions, corresponding to the sensitivity and specificity vectors.
    :param y_sens: sensitivity vector corresponding to the external data proportions specified by x.
    :param y_spec: specificity vector corresponding to the external data proportions specified by x.
    '''

    x_itn, sens_itn = 0, 0
    min_diff = 100.
    for i in range(x.shape[0]):
        abs_diff = np.abs(y_sens[i] - y_spec[i])
        if abs_diff < min_diff:
            min_diff = abs_diff
            x_itn = x[i]
            sens_itn = y_sens[i]
            spec_itn = y_spec[i]
    return x_itn, sens_itn, spec_itn


def export_mmode_to_png(npz,npz_dir,save_dir):
    '''
    Loads an mmode from an npz file and saves it as a png.
    :param npz: file name of the npz (with file extension)
    :param npz_dir: directory where the npz is stored
    :param save_dir: directory where you'd like the png saved
    '''
    i = np.load(os.path.join(npz_dir, npz))['mmode'].astype(np.uint8)
    plt.imshow(i, cmap='gray')
    plt.savefig(os.path.join(save_dir,npz.split('.')[0]) + ".png")


def save_mmodes():
    '''
    Converts external mmode npzs to pngs. Npz directory must be organized by center, then class.
    '''
    npz_root = cfg['EXTERNAL_VAL']['PATHS']['NPZS']
    save_root = cfg['EXTERNAL_VAL']['PATHS']['PNGS']
    for center in cfg['EXTERNAL_VAL']['LOCATIONS']:
        for label in ['sliding', 'no_sliding']:
            npz_path = os.path.join(npz_root, center, label)
            save_path = os.path.join(save_root, center, label)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
                print('new dir created at {}'.format(save_path))
            for root, dir, files in os.walk(npz_path):
                print(len(files))
                for file in files:
                    if file.split("_")[1] == '1.npz': # Only save the first (brightest pixel) m-mode for each clip
                        export_mmode_to_png(file,npz_path,save_path)


if __name__ == '__main__':
    exp_dir = os.path.join(os.getcwd() + cfg['PATHS']['EXPERIMENTS'])
    exp_path = os.path.join(exp_dir, os.listdir(exp_dir)[-1])

    # Plot overall results
    plot_finetune_results(exp_path, flip_labels=False)

    # Plot results by probe type
    plot_subgroup_analysis_results(exp_path, flip_labels=False, group='probe')

    # Plot mean sensitivity and specificity
    plot_mean_sensitivity_specificity_tradeoff(exp_path, flip_labels=False, print_metrics=True)




