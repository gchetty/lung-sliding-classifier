# Lung Sliding Classifier
![Deep Breathe Logo](img/readme/deep-breathe-logo.jpg "Deep Breath AI")   

We at [Deep Breathe](https://www.deepbreathe.ai/) sought to train a deep learning model for the task
of predicting the presence (positive class) or absence (negative class) of lung sliding, for the purpose of automatic pneumothorax detection from lung ultrasound videos.

## Table of Contents
1. [**_Getting Started_**](#getting-started)
2. [**_Building a Dataset_**](#building-a-dataset)  
    i)[**_Additional data_**](#additional-data)
3. [**_Model Training_**](#model-training)
4. [**_Contacts_**](#contacts)

## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Install the necessary dependencies (listed in
   [requirements.txt](requirements.txt)) in a new Python virtual environment. To do this in Windows (minor differences for Mac / Linux), open a terminal in
   the root directory of the project and run the following:
   ```
   > Python -m venv .env
   > .env\Scripts\activate 
   (.env) > pip install -r requirements.txt
   ```  
3. Obtain lung ultrasound data and preprocess it accordingly. See
   [building a dataset](#building-a-dataset) for more details.
4. Update the _`TRAIN` >> `MODEL_DEF`_ field of [_config.yml_](config.yml) with
   the appropriate string representing the model type you wish to
   run, along with any other desired parameters. To train a model, ensure the `TRAIN` >>
   `EXPERIMENT_TYPE` field is set to _'train_single'_. (Note: Anywhere the style of `X` >> `Y` etc. is used, we are 
   referring to the corresponding parameter in the [_config.yml_](config.yml) file) See [model training](#model-training) for details on this section.
5. Execute [_make_splits.py_](src/make_splits.py), then [_train.py_](src/train.py) to train your chosen model on your
   preprocessed data. The trained model will be saved in the path specified by `TRAIN` >> `PATHS` >> `MODEL_OUT`. Its filename will resemble the following
   structure: _model{yyyymmdd-hhmmss}_, where _{yyyymmdd-hhmmss}_ is the current date and
   time.
6. Navigate to `TRAIN` >> `PATHS` >> `TENSORBOARD` to see the tensorboard log files, which can be used to create a [tensorboard](https://www.tensorflow.org/tensorboard)
   visualization of the training results. See the resultant saved model in `TRAIN` >> `PATHS` >> `MODEL_OUT`.
   
## Building a Dataset
1. Set the parameters under _`PREPROCESS` >> `PARAMS`_ in [_config.yml_](config.yml) as desired.
2. The following CSVs must be placed in the _`PREPROCESS` >> `PATHS` >> `CSVS_OUTPUT`_: 
{class}_bad_ids.csv, {class}_boxes.csv, sliding_extra.csv, where {class} is 'sliding' and 'no_sliding'. These csvs
contain  clip ID's to exclude, pleural region ROIs, and extra clips for the sliding class. These csvs can be 
downloaded from the private Deep Breathe Google Drive (Deep Breathe -> lung_sliding_classifier -> data -> 
new_csvs). 
3. Run [_run.py_](src/data/run.py). This will generate M-mode images in NPZ format, as required
for the training pipeline.

When running [_run.py_](src/data/run.py), the following steps are run:
1. [_run.py_](src/data/run/py) will first call [_download_videos.py_](src/data/download_videos.py) to download 
appropriate videos from the database.
2. [_mask.py_](src/data/mask.py) is called to apply the masking tool for artifact removal. The raw ultrasound clips are 
scrubbed of all on-screen information e.g. vendor logos, battery indicators, index mark, depth markers)
extraneous to the ultrasound beam itself. This was done using a dedicated
deep learning masking software for ultrasound (AutoMask, WaveBase Inc.,
Waterloo, Canada). 
3. There is then an optional smoothing step, 
[_smoothing_filter.py_](src/data/smoothing_filter.py), which applies a median filter to masked
clips. 
4. [_to_npz.py_](src/data/to_npz.py) will be called to convert masked (and optionally smoothed) 
clips to sub-clips with fixed length, construct M-mode images for each sub-clip, and save the M-mode images in the NPZ 
(serialized NumPy array) format. An NPZ table is generated linking each M-mode to its ground truth, associated 
clip, and patient. This can be found _`PREPROCESS` >> `PATHS` >> `CSVS_OUTPUT`_.

Note: Some additional preprocessing methods exist from early experimentation for this project. This includes using 
optical flow instead of M-Mode images, which can be done by setting `PREPROCESS` >> `PARAMS` >> `FLOW`.

### Additional data

Additional data can also be added to existing internal data for 
training purposes. To do so, scripts in the [_extras_](src/data/extras/) folder containing the 
word 'extra' in their file name will be required, and must be moved to the [_data folder_](src/data/). 
Next, ensure that the previous section addressing internal data has been properly executed. 
A CSV containing pleural region ROIs for these external clips, which can be found in the same 
place as those for internal data (see previous section), is also needed. 

There is currently no unified 'run' script for additional data, so individual scripts will have 
to be called when needed. First, ensure that all additional videos are in the 
[_extra_unmasked_videos_](src/data/extra_unmasked_videos/) directory, and call
[_extra_get_frame_rates.py_](src/data/extras/extra_get_frame_rates.py), which generates frame 
rate CSVs for all additional videos. Next, [_mask_extra.py_](src/data/extras/mask_extra.py) 
masks additional videos. Finally, [_extra_clips_to_npz.py_](src/data/extras/extra_clips_to_npz.py)
should be called to produce mini-clips (with associated metadata) as NPZs. Note that smoothing 
(median filtering) as a preprocessing step is not currently implemented for additional data. 

## Use Cases

### Model Training
To train a model whose architecture is already defined in this repository:
1. Assemble a pre-processed M-mode dataset (see [**_Building a Dataset_**](#building-a-dataset)). 
2. Run [_make_splits.py_](src\make_splits.py) to generate proper train / validation / test splits, where the 
proportions can be adjusted in the _`TRAIN` >> `SPLITS`_ section of 
[config.yml](config.yml). Additionally, if additional data is to be used, run 
[_add_extra_to_splits_](src/extras/add_extra_to_splits.py) to add to existing splits.
3. Update the _`TRAIN` >> `MODEL_DEF`_ field of [_config.yml_](config.yml) with the appropriate string representing the model type you 
wish to train, and `TRAIN` >> `EXPERIMENT_TYPE` to _single_train_. 
4. Set the associated hyperparamerer values based on the chosen model definition (in `TRAIN` >> `PARAMS`).
5. Run [train.py](src/train.py).
6. View logs and saved models in the directory specified in `TRAIN` >> `PATHS` >> `TENSORBOARD` and `MODEL_OUT`. Logs 
can be viewed in TensorBoard by running `tensorboard --logdir path` in terminal, where path is the directory containing 
the logs.

Note: We found that the EfficientNetB0 model had the best performance on our internal data.

To train a custom model, please follow the style laid out in [models.py](src/models/models.py) and place the code in 
that file. Furthermore, please add to the config file by following the style in _TRAIN >> PARAMS_ to add configurable 
options, such as the learning rate. Although you don't _have_ to make use of these parameters, please at least create 
this section in [config.yml](config.yml) with the same name in capitals as you decided in [models.py](src/models/models.py).

### K-Fold Cross Validation
1. Assemble a pre-processed M-mode dataset (see [**_Building a Dataset_**](#building-a-dataset)). 
2. Run [_make_splits.py_](src\make_splits.py) to generate proper train / validation / test splits, where the 
proportions can be adjusted in `TRAIN` >> `SPLITS`. Note that the K-fold cross validation experiment will combine the 
train and validation splits to partition the data into k-folds, while leaving the test split out. 
Additionally, if additional data is to be used, run 
[_add_extra_to_splits_](src/extras/add_extra_to_splits.py) to add to existing splits.
3. Update the `TRAIN` >> `MODEL_DEF` field of [_config.yml_](config.yml) with the appropriate string representing the model type you 
wish to train, and `TRAIN` >> `EXPERIMENT_TYPE` to _cross_val_. 
4. Set the associated hyperparamerer values based on the chosen model definition (in `TRAIN` >> `PARAMS`).
5. Set the number of folds in `CROSS_VAL` >> `N_FOLDS`, along with other configurable parameters. 
6. Run [train.py](src/train.py).
7. View logs and saved models in the directory specified in `TRAIN` >> `PATHS` >> `TENSORBOARD` and `MODEL_OUT`. Logs 
can be viewed in TensorBoard by running `tensorboard --logdir path` in terminal, where path is the directory containing 
the logs.

Note: The k-fold data partitions can be found in the directory specified in `CROSS_VAL` >> `PARTITIONS`.

### Bayesian Hyperparameter Optimization
1. Assemble a pre-processed M-mode dataset (see [**_Building a Dataset_**](#building-a-dataset)). 
2. Run [_make_splits.py_](src\make_splits.py) to generate proper train / validation / test splits, where the 
proportions can be adjusted in `TRAIN` >> `SPLITS`. Note that the hyperparameter optimization will not use the test set
for model evaluation. Simply generate the train / validation / test splits and only training and validation will be used.
For a hyperparameter optimization with integrated cross-validation experiment, 
train and validation splits will be combined to partition the data into k-folds (as set in `CROSS_VAL` >> `N_FOLDS`), while leaving 
the test split out.
If additional data is to be used, run 
[_add_extra_to_splits_](src/extras/add_extra_to_splits.py) to add to existing splits.
3. Update the `TRAIN` >> `MODEL_DEF` field of [_config.yml_](config.yml) with the appropriate string representing the model type you 
wish to train, and `TRAIN` >> `EXPERIMENT_TYPE` to _hparam_search_ or  _hparam_cross_val_search_.
4. Set the associated model hyperparameter values based on the chosen model definition (in `TRAIN` >> `PARAMS`).
5. Set the hyperparameter types and ranges to vary for optimization in `HPARAM_SEARCH` >> `MODEL_DEF`, where _MODEL_DEF_ 
is the same model chosen in `TRAIN` >> `MODEL_DEF`.
6. Set the number of runs to complete for the optimization in `HPARAM_SEARCH` >> `N_EVALS`, as well as other configurable
parameters. 
Note that for integrated cross validation, the total number of training runs will be `N_EVALS` x `CROSS_VAL_RUNS`.
7. Run [train.py](src/train.py).
8. View logs and saved models in the directory specified in `TRAIN` >> `PATHS` >> `TENSORBOARD` and `MODEL_OUT`. Logs 
can be viewed in TensorBoard by running `tensorboard --logdir path` in terminal, where path is the directory containing 
the logs.

Note: A CSV file of experiment results and a partial dependance plot can be found in `HPARAM_SEARCH` >> `PATH`.
The k-fold data partitions for cross validation can be found in the directory specified in `CROSS_VAL` >> `PARTITIONS`.

### Predictions
With a trained model, you can compute M-mode predictions and metrics on a dataset using the following steps:
1. Assemble a pre-processed M-mode dataset to generate predictions on (see [**_Building a Dataset_**](#building-a-dataset)). 
2. Set `PREDICT` >> `MODEL` to point to a trained model.
3. Set `PREDICT` >> `TEST_DF` to point to an NPZ table linking each M-mode to its ground truth, associated clip, and patient.
4. Execute [_predict.py_](src/predict.py). Predictions and metrics are outputted to `PREDICT` >> `PREDICTIONS_OUT`.

### Grad-CAM Explainability Heatmaps for M-Mode Images
With a trained model and a collection of preprocessed clips (converted to M-modes), you can apply Grad-CAM visualization.
1. Assemble a pre-processed M-mode dataset to generate heatmaps for (see [**_Building a Dataset_**](#building-a-dataset)).
2. Set `EXPLAINABILITY` >> `PATHS` >> `MODEL` to point to a trained model.
3. To generate a heatmap on a single M-mode image, set `EXPLAINABILITY` >> `SINGLE_MMODE` to _True_. To generate heatmaps for 
multiple M-modes from a dataset, set this to _True_.
4. Set `EXPLAINABILITY` >> `PATHS` >> `NPZ` and `NPZ_DF` to point to the directories holding the preprocessed NPZ 
files and the NPZ table linking each M-mode to its ground truth, associated clip, and patient.
5. Execute [_gradcam.py_](src/gradcam.py). Ouput heatmaps can be found in `EXPLAINABILITY` >> `PATHS` >> `FEATURE_HEATMAPS`.

## Project Configuration
This project contains several configurable variables that are defined in
the project config file: [config.yml](config.yml). When loaded into
Python scripts, the contents of this file become a dictionary through
which the developer can easily access its members.

For user convenience, the config file is organized into major components
of the model development pipeline. Many fields do not need to be modified by
the typical user, but others may be modified to suit the user's specific
goals.

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as their only purpose is to force Git to track
empty directories.

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
│   ├── custom
│   |   └── metrics.py               <- Script containing custom metrics
│   ├── data
|   |   ├── extras                   <- Old scripts for pulling and processing additional data
|   |   ├── denseflow.py             <- Script for extracting optical flow images from video frames
|   |   ├── auto_masking.py          <- Script for running WaveBase Inc. Auto-Masking software - see https://www.wavebase.ai/
|   |   ├── download_videos.py       <- Script for pulling mp4 clips and metadata from the cloud
|   |   ├── flow.py                  <- Script for setting up and running denseflow.py
|   |   ├── mask.py                  <- Script for setting up and running auto_masking.py
|   |   ├── run.py                   <- Script for running all required scripts for the data pipeline
|   |   ├── smoothing_filter.py      <- Script for applying frame-wise smoothing median filter to clips
|   |   ├── to_npz.py                <- Script for converting clips to M-mode images and storing them in NPZ format
|   |   └── utils.py                 <- Script containing commonly used utility functions
|   ├── extras
|   |   └── add_extra_to_splits.py   <- Script for adding additional negatives to train/test/val splits
│   ├── models                       
|   |   └── models.py                <- Script containing all model definitions
│   ├── visualization
|   |   └── vizualization.py         <- Script for generating various plots and graphs
|   ├── explanations.py              <- Script for generating 3D Grad-CAM heatmaps from clips
|   ├── gradcam.py                   <- Script containing Grad-CAM application and heatmap generation
|   ├── make_splits.py               <- Script for generating train/val/test splits from data
|   ├── predict.py                   <- Script for prediction on M-modes using trained models
|   ├── preprocessor.py              <- Script for preprocessing data (NPZ files) and preparing it for model input
|   └── train.py                     <- Script for training experiments
|
├── .gitignore                       <- Files to be be ignored by git.
├── config.yml                       <- Values of several constants used throughout project
├── README.md                        <- Project description
└── requirements.txt                 <- Lists all dependencies and their respective versions
```

## Contacts

**Robert Arntfield**  
Project Lead  
Deep Breathe  
robert.arntfield@gmail.com

**Blake VanBerlo**  
Deep Learning Project Lead   
Deep Breathe  
bvanberlo@uwaterloo.ca
