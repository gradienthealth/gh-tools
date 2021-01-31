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

    def get_config(self):
        return {"name": "normalize"}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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

    def get_config(self):
        return {"name": "denormalize"}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PaddingLayer(tf.keras.layers.Layer):
    def __init__(self, divisible=16, name='padding'):
        super().__init__()
        self.divisible = divisible
        
    def call(self, inputs):
        shape = tf.shape(inputs)
        divisible = tf.constant(self.divisible, tf.int32)
        pady = (divisible - shape[1] % divisible)
        padx = (divisible - shape[2] % divisible)
        y1 = pady//2
        y2 = pady - y1
        x1 = padx//2
        x2 = padx - x1       
        paddings = tf.stack([
          tf.stack([0,0], axis=0),
          tf.stack([y1, y2], axis=0),
          tf.stack([x1, x2], axis=0),
          tf.stack([0,0], axis=0),
        ], axis=0)

        x = inputs
        x = tf.pad(x, paddings, "REFLECT")
        return x, y1, y2, x1, x2
    def get_config(self):
        return {"divisible": self.divisible}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DepaddingLayer(tf.keras.layers.Layer):
    def __init__(self, name='padding'):
        super().__init__()

    def call(self, inputs):
        x = inputs[0]
        shape = tf.shape(x)
        padyl = inputs[1]
        padyr = shape[1] - inputs[2]
        padxl = inputs[3]
        padxr = shape[2] - inputs[4]
        x = x[:, padyl:padyr, padxl:padxr, :]
        return x

    def get_config(self):
        return {"name": "depad"}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
