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
        self.divisible = tf.constant(divisible, tf.int32)
        
    def call(self, inputs):
        shape = tf.shape(inputs)
        pady = (self.divisible - shape[1] % self.divisible)
        padx = (self.divisible - shape[2] % self.divisible)
        padyl = pady//2
        padyr = pady - padyl
        padxl = padx//2
        padxr = padx - padxl       
        paddings = tf.stack([
          tf.concat([0,0], axis=0),
          tf.concat([y1, y2], axis=0),
          tf.concat([x1, x2], axis=0),
          tf.concat([0,0], axis=0),
        ], axis=0)

        x = inputs
        x = tf.pad(x, paddings, "REFLECT")
        return x, padyl, padyr, padxl, padxr
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
