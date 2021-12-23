[comment]: <> (TODO: Add project name.)
# Lung Sliding Classifier
![Deep Breathe Logo](img/readme/deep-breathe-logo.jpg "Deep Breath AI")   

[comment]: <> (TODO: Add project description.)
We at [Deep Breathe](https://www.deepbreathe.ai/) sought to train a deep learning model for the task
of predicting the presence (positive class) or absence (negative class) of lung sliding, for the purpose of automatic pneumothorax detection from lung ultrasound videos.

[comment]: <> (TODO: Update table of contents to use correct links and section titles.)
## Table of Contents
1. [**_Getting Started_**](#getting-started)
2. [**_Building a Dataset_**](#building-a-dataset)
   i)[**_Extraneous data_**](#extraneous-data)
3. [**_Model Training_**](#model-training)
4. [**_Contacts_**](#contacts)

[comment]: <> (TODO: Update the getting started section to reflect the project's specific setup.)
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
   [building a dataset](#building-a-dataset) for more details, but a simple terminal call of `Python run.py` in src/data/ _should_ be enough.
4. Update the _TRAIN >> MODEL_DEF_ field of [_config.yml_](config.yml) with
   the appropriate string representing the model type you wish to
   train, along with any other desired parameters (Note: Anywhere the style of _X >> Y_ etc. is used, we are referring to the corresponding parameter in the config.yml file) See [model training](#model-training) for details on this section.
5. Execute [make_splits.py](src/make_splits.py), then [_train.py_](src/train.py) to train your chosen model on your
   preprocessed data. The trained model will be saved in the path specified by _TRAIN >> PATHS >> MODEL_OUT_.
6. Navigate to _TRAIN >> PATHS >> TENSORBOARD_ to see the tensorboard log files, which can be used to create a [tensorboard](https://www.tensorflow.org/tensorboard)
   visualization of the training results. See the resultant saved model in _TRAIN >> PATHS >> MODEL_OUT_. 
   
   
## Building a Dataset

For all clips contained in our internal database, a single call to [_run.py_](src/data/run.py)
will generate mini-clips in NPZ (serialized NumPy array) form, as required for the training pipeline. Before running this
file, ensure that the parameters under _PREPROCESS >> PARAMS_ in [_config.yml_](config.yml) are 
set as desired. Additionally, CSVs containing pleural region ROIs are needed, and can be 
downloaded from the appropriate source (Deep Breathe -> lung_sliding_classifier -> data -> 
new_csvs -> Boxes). [run.py](src/data/run/py) will first call [_download_videos.py_](src/data/download_videos.py) to download appropriate videos from the database, then [_mask.py_](src/data/mask.py) is 
called to apply the masking tool for artifact removal. There is then an optional smoothing step, 
[_smoothing_filter.py_](src/data/smoothing_filter.py), which applies a median filter to masked
clips. Finally, [_to_npz.py_](src/data/to_npz.py) will be called to convert masked (and optionally smoothed) 
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


## Model Training
To train a model that is already defined in this repository, first run [make_splits.py](src\make_splits.py) to generate proper train / validation / test splits, where the proportions can be adjusted in the _TRAIN >> SPLITS_ section of [config.yml](config.yml). Afterwards, update the _TRAIN >> MODEL_DEF_ field of [_config.yml_](config.yml) with the appropriate string representing the model type you wish to train, where the list of current options is in [models.py](src/models/models.py). To train the model on those splits, simply run [train.py](src/train.py). However, there are _many_ optional parameters in [config.yml](config.yml) - please consult that file to understand the full list of options. In particular, one notable option is whether or not to apply an M Mode transformation, which when paired with conventional CNNs such as Xception, shows the most promise at the time of writing.

To train a custom model, please follow the style laid out in [models.py](src/models/models.py) and place the code in that file. Furthermore, please add to the config file by following the style in _TRAIN >> PARAMS_ in the config file to add configurable options, such as the learning rate. Although you don't _have_ to make use of these parameters, please at least create this section in [config.yml](config.yml) with the same name in capitals as you decided in [models.py](src/models/models.py).

To observe the training results of the model in TensorBoard, run @Brian (haven't done the tensorboard stuff in months sorry). The saved model will be located in [models/model_name](src/results/model/). 

## Contacts

**Robert Arntfield**  
Project Lead  
Deep Breathe  
robert.arntfield@gmail.com

**Blake VanBerlo**  
Deep Learning Project Lead   
Deep Breathe  
bvanberlo@uwaterloo.ca
