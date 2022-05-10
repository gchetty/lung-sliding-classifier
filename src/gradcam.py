import os

import yaml
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tkinter import filedialog as fd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from src.visualization.visualization import visualize_heatmap
from src.models.models import get_model
from src.preprocessor import MModePreprocessor
from src.predict import predict_set
from tqdm import tqdm

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../config.yml"), 'r'))


class GradCAMExplainer:

    def __init__(self):
        self.model = load_model(os.path.join('..', cfg['EXPLAINABILITY']['PATHS']['MODEL']), compile=False)
        self.save_img_dir = cfg['EXPLAINABILITY']['PATHS']['FEATURE_HEATMAPS']
        self.npz_dir = cfg['EXPLAINABILITY']['PATHS']['NPZ']
        self.img_dim = tuple((cfg['PREPROCESS']['PARAMS']['M_MODE_WIDTH'], cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0]))
        self.classes = ['Absent Lung Sliding', 'Lung Sliding']
        self.x_col = 'filename'
        self.y_col = 'label'
        _, preprocessing_fn = get_model(cfg['TRAIN']['MODEL_DEF'])
        self.preprocessing_fn = preprocessing_fn

        # Make sure heatmap directory exists
        if not os.path.exists(self.save_img_dir):
            os.makedirs(self.save_img_dir)

        # Get name of final convolutional layer
        layer_name = ''
        for layer in self.model.layers:
            if any('Conv' in l for l in layer._keras_api_names):
                layer_name = layer.name
        self.last_conv_layer = layer_name
        self.hm_intensity = 0.5

    def apply_gradcam(self, npz_df):
        '''
        For each image in the dataset provided, make a prediction and overlay a heatmap depicting the gradient of the
        predicted class with respect to the feature maps of the final convolutional layer of the model.
        :param npz_df: Pandas Dataframe of npz files, linking npz filenames to labels
        '''

        for i in ['sliding', 'no_sliding']:
            if not os.path.exists(os.path.join(self.save_img_dir, i)):
                os.makedirs(os.path.join(self.save_img_dir, i))

        # get dataset and apply preprocessor
        ds = tf.data.Dataset.from_tensor_slices((npz_df['filename'].tolist(), npz_df['label']))
        preprocessor = MModePreprocessor(self.preprocessing_fn)
        preprocessed_set = preprocessor.prepare(ds, npz_df, shuffle=False, augment=False)

        preds, probs = predict_set(self.model, npz_df)
        idx = 0
        for x, y in tqdm(preprocessed_set):
            filename = npz_df['filename'].iloc[idx]
            # Obtain gradient of output with respect to last convolutional layer weights
            with tf.GradientTape() as tape:
                last_conv_layer = self.model.get_layer(self.last_conv_layer)
                iterate = Model([self.model.inputs], [self.model.output, last_conv_layer.output])
                model_out, last_conv_layer = iterate(x)
                class_out = tf.math.maximum(1 - model_out[:, np.argmax(model_out[0])],
                                            model_out[:, np.argmax(model_out[0])])
                grads = tape.gradient(class_out, last_conv_layer)
                pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))

            # Upsample and overlay heatmap onto original image
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
            heatmap = np.array(heatmap).squeeze(axis=0)
            heatmap = np.maximum(heatmap, 0.0)  # Equivalent of passing through ReLU
            heatmap /= np.max(heatmap)
            heatmap = cv2.resize(heatmap, self.img_dim)
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            orig_img = np.load(filename)['mmode']
            orig_img = cv2.convertScaleAbs(cv2.cvtColor(np.float32(orig_img), cv2.COLOR_GRAY2RGB))
            heatmap_img = cv2.addWeighted(heatmap, self.hm_intensity, orig_img, 1.0 - self.hm_intensity, 0)

            # Visualize the Grad-CAM heatmap and optionally save it to disk
            img_filename = npz_df[self.x_col].iloc[idx]
            label = npz_df[self.y_col].iloc[idx]
            folder = 'sliding' if label == 1 else 'no_sliding'
            _ = visualize_heatmap(orig_img, heatmap_img, img_filename, label, probs[idx], self.classes,
                                  dir_path=os.path.join(self.save_img_dir, folder))
            idx += 1
        return heatmap

    def get_single_heatmap(self, npz_df=None):
        '''
        Apply Grad-CAM to an individual M-Mode
        :param frame_df: Pandas DataFrame of LUS frames
        :return: The heatmap produced by Grad-CAM
        '''

        file_path = fd.askopenfilename(initialdir=cfg['EXPLAINABILITY']['PATHS']['NPZ'], title='Select a clip',
                                       filetypes=[('npz files', '*.npz')])

        if npz_df is None:
            npz_df = pd.read_csv(cfg['EXPLAINABILITY']['PATHS']['NPZ_DF'])

        filtered_df = npz_df[npz_df['id'] == os.path.basename(file_path).rstrip('.npz')]
        filtered_df.reset_index(inplace=True)

        heatmap = self.apply_gradcam(filtered_df)
        return heatmap


if __name__ == '__main__':
    gradcam = GradCAMExplainer()
    if cfg['EXPLAINABILITY']['SINGLE_MMODE']:
        gradcam.get_single_heatmap()
    else:
        gradcam.apply_gradcam(npz_df=pd.read_csv(cfg['EXPLAINABILITY']['PATHS']['NPZ_DF']))
