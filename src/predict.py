'''
Script for running predictions using trained models
'''

import datetime
import pandas as pd
from src.visualization.visualization import *
from src.models.models import get_model
from preprocessor import MModePreprocessor
from tensorflow.keras.models import load_model

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def predict_set(model, test_df, threshold=0.5, model_def_str=cfg['TRAIN']['MODEL_DEF']):
    '''
    Given a dataset, make predictions for each constituent example.
    :param model: A trained TensorFlow model
    :param preprocessing_func: Preprocessing function to apply before sending image to model
    :param predict_df: Pandas Dataframe of LUS frames, linking image filenames to labels
    :param threshold: Classification threshold
    :param model_def_str: Model name
    :return: List of predicted classes, array of classwise prediction probabilities
    '''

    # Get the model function and preprocessing function
    model_def_fn, preprocessing_fn = get_model(model_def_str)

    # Create TF datasets for training, validation and test sets
    # Note: This does NOT load the dataset into memory! We specify paths,
    #       and labels so that TensorFlow can perform dynamic loading.
    dataset = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label']))
    preprocessor = MModePreprocessor(preprocessing_fn)

    preprocessed_set = preprocessor.prepare(dataset, test_df, shuffle=False, augment=False)
    # Obtain prediction probabilities and classes
    p = model.predict(preprocessed_set, verbose=1)
    pred_classes = (p[:, 0] >= threshold).astype(int)

    return pred_classes, np.squeeze(p, axis=1)

def export_predictions(df, pred_classes, probs):
    ids = df['id'].to_list()
    pred_df = pd.DataFrame(columns=['id', 'Predicted Class', 'Probability'])
    pred_df['id'] = df['id']
    pred_df['Ground Truth'] = df['label']
    pred_df['Predicted Class'] = pred_classes
    pred_df['Probability'] = probs
    pred_df.to_csv(os.path.join(os.getcwd(),
                                cfg['PREDICT']['PREDICTIONS_OUT'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                + '_predictions.csv'), index=False)


if __name__=='__main__':
    test_df = pd.read_csv(cfg['PREDICT']['TEST_DF'])
    model = load_model(os.path.join('..', cfg['PREDICT']['MODEL']), compile=False)
    pred_classes, probs = predict_set(model, test_df)
    export_predictions(test_df, pred_classes, probs)