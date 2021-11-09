'''
Script for any explainability functions.
Example: ab-classifier used GradCAM
'''

import yaml
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from scipy import ndimage
from ..models.models import *
from tensorflow_addons.metrics import F1Score, FBetaScore
from ..custom.metrics import Specificity
from ..preprocessor import Preprocessor

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))


class ClipGradCam:

    def __init__(self):
        custom_objects = {'Specificity': Specificity, 'FBetaScore': FBetaScore}
        self.model = load_model(cfg['EXPLAINABILITY']['PATHS']['MODEL'], custom_objects=custom_objects)
        self.heatmap_dir = cfg['EXPLAINABILITY']['PATHS']['FEATURE_HEATMAPS']
        self.npz_dir = cfg['EXPLAINABILITY']['PATHS']['NPZ']
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

        # get dataset and apply preprocessor
        miniclip_csv = pd.read_csv(cfg['EXPLAINABILITY']['PATHS']['NPZ_DF'])
        ds = tf.data.Dataset.from_tensor_slices((miniclip_csv['filename'].tolist(), miniclip_csv['label']))
        preprocessor = Preprocessor(self.preprocessing_fn)
        ds = preprocessor.prepare(ds, miniclip_csv, shuffle=False, augment=False)

        # load frame rate csv
        frame_rates_sliding = pd.read_csv(os.path.join(cfg['EXPLAINABILITY']['PATHS']['FR_CSVS']), 'sliding_frame_rates.csv')
        frame_rates_no_sliding = pd.read_csv(os.path.join(cfg['EXPLAINABILITY']['PATHS']['FR_CSVS']), 'no_sliding_frame_rates.csv')
        frame_rates = pd.concat([frame_rates_sliding, frame_rates_no_sliding])

        index = 0
        for clip, label in ds:

            id = miniclip_csv['id'].iloc[index]
            filename = miniclip_csv['filename'].iloc[index]

            with np.load(filename, allow_pickle=True) as c:
                orig_clip = c['frames']

            # get frame rate
            frame_rate = (frame_rates[frame_rates['id'] == id])['frame_rate'].values[0]

            with tf.GradientTape() as tape:
                last_conv_layer = self.model.get_layer(self.last_conv_layer)
                tape.watch(last_conv_layer.variables)
                iterate = tf.keras.models.Model([self.model.inputs], [self.model.output, last_conv_layer.output])
                model_out, last_conv_layer = iterate(clip)
                class_out = model_out
                grads = tape.gradient(class_out, last_conv_layer)
                pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2, 3))

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
            heatmap = np.array(heatmap)
            heatmap = heatmap.squeeze(axis=0)
            heatmap = heatmap.astype(np.float32)
            heatmap = ndimage.zoom(heatmap,
                                   (90.0 / (heatmap.shape[0]), 224.0 / (heatmap.shape[1]), 224.0 / (heatmap.shape[2])))
            heatmap = np.maximum(heatmap, 0.0)
            heatmap /= np.max(heatmap)
            heatmap = np.uint8(255 * heatmap)

            print(heatmap.shape)
            print(orig_clip.shape)

            heatmap_imgs = []
            for hmap, frame in zip(heatmap, orig_clip):
                hmap = cv2.applyColorMap((255 * hmap).astype(np.uint8), cv2.COLORMAP_HOT)
                heatmap_imgs.append(cv2.addWeighted(hmap, self.hm_intensity, frame, 1.0 - self.hm_intensity, 0, dtype=cv2.CV_8U))

            self.save_heatmap_video(heatmap_imgs, id, frame_rate)

            index += 1

    def save_heatmap_video(self, heatmaps, id, fps):
        size = (heatmaps[0].shape[0], heatmaps[0].shape[1])
        out = cv2.VideoWriter(self.heatmap_dir + id + 'heatmap.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for heatmap in heatmaps:
            out.write(heatmap)
        out.release()


if __name__ == '__main__':
    os.chdir('../')  # change working directory to src/
    cam = ClipGradCam()
    cam.apply_gradcam()
