[comment]: <> (TODO: Add project name.)
# Lung Sliding Classifier
![Deep Breathe Logo](img/readme/deep-breathe-logo.jpg "Deep Breath AI")   

[comment]: <> (TODO: Add project description.)
We at [Deep Breathe](https://www.deepbreathe.ai/) sought to train a deep learning model for the task
of predicting the presence (positive class) or absence (negative class) of lung sliding.

[comment]: <> (TODO: Update table of contents to use correct links and section titles.)
## Table of Contents
1. [**_Getting Started_**](#getting-started)
2. [**_Building a Dataset_**](#building-a-dataset)
   i)[**_Extraneous data_**](#extraneous-data)
3. [**_Use Cases_**](#use-cases)  
   i)[**_Use Case 1_**](#use-case-1)  
   ii) [**_Use Case 2_**](#use-case-2)  
   iii) [**_Use Case 3_**](#use-case-3)
4. [**_Project Configuration_**](#project-configuration)
5. [**_Project Structure_**](#project-structure)
6. [**_Contacts_**](#contacts)

[comment]: <> (TODO: Update the getting started section to refplect the project's specific setup.)
## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Install the necessary dependencies (listed in
   [requirements.txt](requirements.txt)). To do this, open a terminal in
   the root directory of the project and run the following:
   ```
   $ pip install -r requirements.txt
   ```
[comment]: <> (TODO: Update the data type used for the project.)
3. Obtain _some data type_ data and preprocess it accordingly. See
   [building a dataset](#building-a-dataset) for more details.
   
[comment]: <> (TODO: Update any specific steps, configuration, or directories.)
4. Update the _TRAIN >> MODEL_DEF_ field of [_config.yml_](config.yml) with
   the appropriate string representing the model type you wish to
   train. To train a model, ensure the _TRAIN >>
   EXPERIMENT_TYPE_ field is set to _'train_single'_.
5. Execute [_train.py_](src/train.py) to train your chosen model on your
   preprocessed data. The trained model will be serialized within
   _results/models/_, and its filename will resemble the following
   structure: _model{yyyymmdd-hhmmss}.h5_, where _{yyyymmdd-hhmmss}_ is the current
   time.
6. Navigate to _results/logs/_ to see the tensorboard log files. The folder name will
   be _{yyyymmdd-hhmmss}_.  These logs can be used to create a [tensorboard](https://www.tensorflow.org/tensorboard)
   visualization of the training results.
   
## Building a Dataset

For all clips contained in our internal database, a single call to [_run.py_](src/data/run.py)
will generate mini-clips in npz form, as required for the input pipeline. Before running this
file, ensure that the parameters under preprocess -> params in [_config.yml_](config.yml) are 
set as desired. Additionally, CSVs containing pleural region ROIs are needed, and can be 
downloaded from the appropriate source (Deep Breathe -> lung_sliding_classifier -> data -> 
new_csvs -> Boxes). Specifically, [_download_videos.py_](src/data/download_videos.py) is first 
called to download appropriate videos from the database, and [_mask.py_](src/data/mask.py) is 
called to apply the masking tool for artifact removal. There is then an optional smoothing step, 
[_smoothing_filter.py_](src/data/smoothing_filter.py), which applies a median filter to masked
clips. Finally, [_to_npz.py_](src/data/to_npz.py) converts masked (and optionally smoothed) 
clips to mini-clips with fixed length, saved in the NPZ format. 

### Extraneous data

Data extraneous to the internal database can also be added to existing internal data for 
training purposes. To do so, scripts in the [_extras_](src/data/extras/) folder containing the 
word 'extra' in their file name will be required, and must be moved to the [_data folder_](src/data/). 
Next, ensure that the previous section addressing internal data has been properly executed. 
A CSV containing pleural region ROIs for these external clips, which can be found in the same 
place as those for internal data (see previous section), is also needed. 

There is currently no unified 'run' script for extraneous data, so individual scripts will have 
to be called when needed. First, ensure that all extraneous videos are in the 
[_extra_unmasked_videos_](src/data/extra_unmasked_videos/) directory, and call
[_extra_get_frame_rates.py_](src/data/extras/extra_get_frame_rates.py), which generates frame 
rate CSVs for all extraneous videos. Next, [_mask_extra.py_](src/data/extras/mask_extra.py) 
masks extraneous videos. Finally, [_extra_clips_to_npz.py_](src/data/extras/extra_clips_to_npz.py)
should be called to produce mini-clips (with associated metadata) as NPZs. Note that smoothing 
(median filtering) as a preprocessing step is not currently implemented for extraneous data. 

[comment]: <> (TODO: Add steps to create a data set for model trinaing.)
[comment]: <> (TODO: Include links to scripts used for generating datasets.)
   
## Use Cases

[comment]: <> (TODO: Add project use cases and steps to execute.)

### Use Case 1

With a pre-processed dataset, you can _insert use case name_.

1. 

### Use Case 2

With a pre-processed dataset, you can _insert use case name_.
1. 

### Use Case 3

With a pre-processed dataset, you can _insert use case name_.
1. 

## Project Configuration
This project contains several configurable variables that are defined in
the project config file: [config.yml](config.yml). When loaded into
Python scripts, the contents of this file become a dictionary through
which the developer can easily access its members.

For user convenience, the config file is organized into major components
of the model development pipeline. Many fields need not be modified by
the typical user, but others may be modified to suit the user's specific
goals. A summary of the major configurable elements in this file is
below.
<details closed> 

[comment]: <> (TODO: Update the configuration fields to match config.yml and add any nessesary descriptions.)
[comment]: <> (Note: The configuration fields match the configuration example values defined in config.yml.)
[comment]: <> (Note: The following list of configuration values is simply an exaple set of commonly used parameters.)
[comment]: <> (Note: Be sure to update this readme section as you update the parameters in config.yml.)

<summary>Paths</summary>

This section of the config contains all path definitions for reading data and writing outputs.
- **DATA_TABLE**: Data table in csv format.
- **HEATMAPS**
- **LOGS**
- **IMAGES**
- **MODEL_WEIGHTS**
- **MODEL_TO_LOAD**: Trained model in h5 file format.
- **CLASS_NAME_MAP**: Output class indices in pkl format.
- **BATCH_PREDS**
- **METRICS**
- **EXPERIMENTS**
</details>

<details closed> 
<summary>Data</summary>

- **IMG_DIM**: Dimensions for frame resizing.
- **VAL_SPLIT**: Validation split.
- **TEST_SPLIT**: Test split.
- **CLASSES**: A string list of data classes.
</details>

<details closed> 
<summary>Train</summary>

- **MODEL_DEF**: Defines the type of frame model to train. One of {'vgg16', 'mobilenetv2', 'xception', 'efficientnetb7', 'custom_resnetv2', 'cutoffvgg16'}
- **N_CLASSES**: Number of classes/labels.
- **BATCH_SIZE**: Batch size.
- **EPOCHS**: Number of epocs.
- **PATIENCE**: Number of epochs with no improvement after which training will be stopped.
- **MIXED_PRECISION** Toggle mixed precision training. Necessary for training with Tensor Cores.
- **N_FOLDS**: Cross-validation folds.
- **DATA_AUG**: Data augmentation parameters.
  - **ZOOM_RANGE**
  - **HORIZONTAL_FLIP**
  - **WIDTH_SHIFT_RANGE**
  - **HEIGHT_SHIFT_RANGE**
  - **SHEAR_RANGE**
  - **ROTATION_RANGE**
  - **BRIGHTNESS_RANGE**
- **HPARAM_SEARCH**: 
  - **N_EVALS**: Number of iteration in the bayesian hyperparamter search.
  - **HPARAM_OBJECTIVE**: String identifier for the metric to be optimized by bayesian hyperparamter search.
</details>

<details closed>
<summary>Hyperparameters</summary>

Each model type has a list of configurable hyperparameters defined here.
- **MODEL1**
  - **LR**
  - **DROPOUT**
  - **L2_LAMBDA**
  - **NODES_DENSE0**
  - **FROZEN_LAYERS**
- **MODEL2**
  - **LR**
  - **DROPOUT**
  - **L2_LAMBDA**
  - **NODES_DENSE0**
  - **FROZEN_LAYERS**
</details>

<details closed> 
<summary>Hyperparameter Search</summary>


For each model there is a range of values that can be sampled for the hyperparameter search.
The ranges are defined here in the config file. Each hyperparameter has a name, type, and range.
The type dictates how samples are drawn from the range.

For more information on using bayesian hyperparameters, visit the [skopt documentation](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html).
- **MODEL1**
  - **LR**
    - **TYPE**
    - **RANGE**
  - **DROPOUT**
    - **TYPE**
    - **RANGE**
- **MODEL2**
  - **LR**
    - **TYPE**
    - **RANGE**
  - **DROPOUT**
    - **TYPE**
    - **RANGE**

</details>

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as their only purpose is to force Git to track
empty directories. Disregard any _.\__init\__.py_ files, as they are
empty files that enable Python to recognize certain directories as
packages.

[comment]: <> (TODO: Update the project structure to match your project including descriptions.)

```
├── img
|   ├── experiments                  <- Visualizations for experiments
|   ├── heatmaps                     <- Grad-CAM heatmap images
|   └── readme                       <- Image assets for README.md
├── results
|   ├── data                         
|   |   └── partition                <- K-fold cross-validation fold partitions
|   ├── experiments                  <- Experiment results
|   ├── figures                      <- Generated figures
|   ├── models                       <- Trained model output
|   ├── predictions                  <- Prediction output
│   └── logs                         <- TensorBoard logs
├── src
│   ├── data
|   |   └── database_pull.py         <- Script for pulling clip mp4 files from the cloud - step 2 (specific to our setup)
│   ├── explainability
|   |   └── example_explain.py       <- Script containing some explainability method
│   ├── models
|   |   └── models.py                <- Script containing all model definitions
│   ├── visualization
|   |   └── visualize.py             <- Script for visualization production
│   ├── readme_example_folder
|   |   ├── example1.py              <- Script for...
|   |   └── example2.py              <- Script for...
|   ├── predict.py                   <- Script for prediction on raw data using trained models
|   └── train.py                     <- Script for training experiments
|
├── .gitignore                       <- Files to be be ignored by git.
├── config.yml                       <- Values of several constants used throughout project
├── README.md                        <- Project description
└── requirements.txt                 <- Lists all dependencies and their respective versions
```

## Contacts

**Contact Name**  
Title
Deep Breathe  
Email

**Contact Name**  
Title
Deep Breathe  
Email

