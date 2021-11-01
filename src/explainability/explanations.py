'''
Script for any explainability functions.
Example: ab-classifier used GradCAM
'''

import yaml
import os
from tensorflow.keras.models import load_model

from ..models.models import *

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))


class ClipGradCam:

    def __init__(self):
        self.model = load_model(cfg['TRAIN']['PATHS']['MODEL_OUT'], compile=False)
        self.save_img_dir = cfg['TRAIN']['PATHS']['FEATURE_HEATMAPS']
        self.npz_dir = cfg['PREPROCESS']['PATHS']['NPZ']
        self.img_dim = tuple(cfg['PREPROCESS']['PARAMS']['IMG_SIZE'])
        self.classes = ['No sliding', 'Sliding']
        _, self.preprocessing_fn = get_model(cfg['TRAIN']['MODEL_DEF'])
        self.hm_intensity = 0.5

        # Get name of final convolutional layer
        layer_name = ''
        for layer in self.model.layers:
            if 'conv' in layer.name.lower():
                layer_name = layer.name
        self.last_conv_layer = layer_name

    def apply_gradcam(self):
        print()

if __name__ == '__main__':
    pass