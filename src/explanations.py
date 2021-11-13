import yaml
import os
import cv2
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from tensorflow.keras.models import load_model
from models.models import *
from preprocessor import Preprocessor

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../config.yml"), 'r'))


class GradCAM3D:

    '''
    Applies Grad-CAM and saves resulting heatmap frames as videos for 3D CNNs ONLY
    Necessary components:
        3D CNN in Tensorflow SavedModel format, in directory specified in config file
        NPZ clips (30 FPS) in directory specified in config file
        CSV containing 'id', 'label' (0 or 1) and 'filename' (full path) of NPZ clips to extract heatmaps from
    '''

    def __init__(self):
        model_path = cfg['EXPLAINABILITY']['PATHS']['MODEL']
        self.model = load_model(model_path, compile=False)
        self.heatmap_dir = cfg['EXPLAINABILITY']['PATHS']['FEATURE_HEATMAPS']
        self.npz_dir = cfg['EXPLAINABILITY']['PATHS']['NPZ']
        self.img_dim = tuple(cfg['PREPROCESS']['PARAMS']['IMG_SIZE'])
        self.window = cfg['PREPROCESS']['PARAMS']['WINDOW']
        self.classes = ['No sliding', 'Sliding']
        _, self.preprocessing_fn = get_model(cfg['TRAIN']['MODEL_DEF'])
        self.hm_intensity = 0.5

        # Make sure heatmap directory exists and is empty
        if os.path.exists(self.heatmap_dir):
            shutil.rmtree(self.heatmap_dir)
        os.makedirs(self.heatmap_dir)

        # Get name of final convolutional layer
        layer_name = ''
        for layer in self.model.layers:
            if 'inflated' in model_path:
                if '_conv' in layer.name.lower():
                    layer_name = layer.name
            else:
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
        self.last_conv_layer = layer_name

    def apply_gradcam(self):

        '''
        Extracts and saves heatmaps for all listed mini-clips using NPZ CSV, with path specified in config file
        '''

        # get dataset and apply preprocessor
        miniclip_csv = pd.read_csv(cfg['EXPLAINABILITY']['PATHS']['NPZ_DF'])
        ds = tf.data.Dataset.from_tensor_slices((miniclip_csv['filename'].tolist(), miniclip_csv['label']))
        preprocessor = Preprocessor(self.preprocessing_fn)
        ds = preprocessor.prepare(ds, miniclip_csv, shuffle=False, augment=False)

        index = 0
        for clip, label in ds:

            id = miniclip_csv['id'].iloc[index]
            filename = miniclip_csv['filename'].iloc[index]

            with np.load(filename, allow_pickle=True) as c:
                orig_clip = c['frames']

            # all mini-clips currently downsampled to 30 FPS
            frame_rate = cfg['EXPLAINABILITY']['OUTPUT_FPS']

            # using given input clip, pool gradients at final conv layer, giving a single float per filter
            with tf.GradientTape() as tape:
                last_conv_layer = self.model.get_layer(self.last_conv_layer)
                tape.watch(last_conv_layer.variables)
                iterate = tf.keras.models.Model([self.model.inputs], [self.model.output, last_conv_layer.output])
                model_out, last_conv_layer = iterate(clip)
                class_out = model_out
                grads = tape.gradient(class_out, last_conv_layer)
                pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2, 3))

            # Incorporate filter relevance, remove filter dimension, and resize heatmap
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
            heatmap = np.array(heatmap)
            heatmap = heatmap.squeeze(axis=0)
            heatmap = heatmap.astype(np.float32)
            heatmap = ndimage.zoom(heatmap,
                                   (self.window / (heatmap.shape[0]), self.img_dim[0] / (heatmap.shape[1]), self.img_dim[1] / (heatmap.shape[2])))
            heatmap = np.maximum(heatmap, 0.0)
            heatmap /= np.max(heatmap)
            heatmap = np.uint8(255 * heatmap)

            # Add colour to heatmap images (per-frame), and superimpose heatmaps onto original unaltered frames
            # From lowest to highest focus: Black, red, orange, yellow, white
            heatmap_imgs = []
            for hmap, frame in zip(heatmap, orig_clip):
                hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_HOT)
                heatmap_imgs.append(cv2.addWeighted(hmap, self.hm_intensity, frame, 1.0 - self.hm_intensity, 0, dtype=cv2.CV_8U))

            # Save heatmap frames as a single video
            self.save_heatmap_video(heatmap_imgs, id, frame_rate)

            index += 1

    def save_heatmap_video(self, heatmaps, id, fps):

        '''
        Saves a list of frame-wise heatmaps as a video (mp4)

        :param heatmaps: List of heatmaps, one for each frame
        :param id: ID of the corresponding LUS mini-clip
        :param fps: Frame rate of the outputted video
        '''

        size = (heatmaps[0].shape[0], heatmaps[0].shape[1])
        out = cv2.VideoWriter(self.heatmap_dir + id + '_heatmap.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for heatmap in heatmaps:
            out.write(heatmap)
        out.release()


if __name__ == '__main__':
    cam = GradCAM3D()
    cam.apply_gradcam()
