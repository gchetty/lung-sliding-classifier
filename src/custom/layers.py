import tensorflow as tf

class ToGrayscale(tf.keras.layers.Layer):
    '''
    Converts an RGB image input to a 3-channel grayscale image
    '''

    def __init__(self, name=None):
        super(ToGrayscale, self).__init__(name=name)

    def call(self, inputs):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(inputs))