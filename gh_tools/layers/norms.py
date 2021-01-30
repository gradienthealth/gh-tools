import tensorflow as tf

class NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, name='normalize'):
        super().__init__()

    def call(self, inputs):
        x = inputs
        max = tf.reduce_max(x)
        min = tf.reduce_min(x)
        x = (x-min)/(max-min)
        mean = tf.reduce_mean(x)
        adjusted_stddev = tf.math.maximum(
            tf.math.reduce_std(x), 
            1/tf.math.sqrt(tf.cast(tf.size(x), tf.float32))
        )
        x = (x-mean)/adjusted_stddev
        return [x, min, max, mean, adjusted_stddev]

class DenormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, name='denormalize'):
        super().__init__()

    def call(self, inputs):
        x = inputs[0]
        min = inputs[1]
        max = inputs[2]
        mean = inputs[3]
        std = inputs[4]
        x = (x*std) + mean
        x = x*(max-min) + min
        x = tf.clip_by_value(x, min, max)
        return x
