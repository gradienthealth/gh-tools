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

class BoundLayer(tf.keras.layers.Layer):
    def __init__(self, min_value=0, max_value=1, name='bound'):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()

    def call(self, inputs):
        x = inputs
        curr_min = tf.reduce_min(x)
        curr_max = tf.reduce_max(x)
        x = (x - curr_min) / (curr_max - curr_min)
        dynamic_range = self.max_value - self.min_value
        x = x*dynamic_range + self.min_value
        return x, curr_min, curr_max

    def get_config(self):
        return {
            "max_value": self.max_value,
            "min_value": self.min_value
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DeBoundLayer(tf.keras.layers.Layer):
    def __init__(self, name='debound'):
        super().__init__()

    def call(self, inputs):
        x = inputs[0]
        min_value = tf.reduce_min(x)
        max_value = tf.reduce_max(x)
        x = (x - min_value)/ (max_value - min_value)
        old_min = inputs[1]
        old_max = inputs[2]
        old_dynamic_range = old_max - old_min
        x = x*old_dynamic_range + old_min
        return x

    def get_config(self):
        return {"name": "debound"}

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

class CenterCropping(tf.keras.layers.Layer):
    def __init__(self, size=(512, 512), mode='clip', name='cropping'):
        super().__init__()
        self.size = size
        self.mode = mode
        assert self.mode is 'clip'

    @tf.function
    def call(self, inputs):
        image = inputs
        shape = tf.shape(inputs)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channel = shape[3]
        size = tf.reduce_min(shape) // 2

        if height > width: 
          resize = width if self.mode is 'clip' else height
          image = tf.image.crop_to_bounding_box(image, tf.abs(height-width)//2, 0, resize, resize)
        if width > height: 
          resize = height if self.mode is 'clip' else width
          image = tf.image.crop_to_bounding_box(image, 0, tf.abs(height-width)//2, resize, resize)
        image = tf.image.resize(image, size=self.size, method='bilinear')
        return image, height, width
    def get_config(self):
        return {"size": self.size}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DeCenterCropping(tf.keras.layers.Layer):
    def __init__(self, mode='clip', name='decropping'):
        super().__init__()
        self.mode = mode
        assert self.mode is 'clip'

    @tf.function
    def call(self, inputs):
        image = inputs[0]
        height = inputs[1]
        width = inputs[2]       
        diff = tf.abs(height-width)

        if height >= width:
            resize = width if self.mode is 'clip' else height
            image = tf.image.resize(image, size=(resize, resize), method='bilinear')
            image = tf.pad(image, [[0,0], [diff//2, diff - diff//2], [0,0], [0,0]])
        else:
            resize = height if self.mode is 'clip' else width
            image = tf.image.resize(image, size=(height, height), method='bilinear')
            image = tf.pad(image, [[0,0], [0,0], [diff//2, diff - diff//2], [0,0]])
        return image

    def get_config(self):
        return {"name": "decropping"}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Logit2ProbabilityLayer(tf.keras.layers.Layer):
    def __init__(self, name='logit2probability'):
        super().__init__()

    def call(self, inputs):
        x = inputs
        return tf.math.exp(x)/(1 + tf.math.exp(x))

    def get_config(self):
        return {"name": "logit2probability"}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, name='resize', size=None):
        assert size is not None, 'provide an image size such as (224, 224)'
        self.size = size
        super().__init__()

    def call(self, inputs):
        return [tf.image.resize(inputs, self.size), tf.shape(inputs)[1], tf.shape(inputs)[2]]

    def get_config(self):
        return {"size": self.size}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DeResizeLayer(tf.keras.layers.Layer):
    def __init__(self, name='Deresize'):
        super().__init__()

    def call(self, inputs):
        height = inputs[1]
        width = inputs[2]
        return tf.image.resize(inputs[0], (height, width))

    def get_config(self):
        return {"name": "Deresize"}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
