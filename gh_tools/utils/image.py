import tensorflow as tf
import tensorflow_addons as tfa

@tf.function
def rotational_matrix(t):
  '''t is the counter clockwise rotational angle in radians on a -1 to 1 coordinate system'''
  t = tf.cast(t, tf.float32)
  a = tf.math.cos(t)
  b = tf.math.sin(t)
  m = tf.stack([a, -b, 0, b, a, 0, 0, 0, 1])
  m = tf.reshape(m, [3,3])
  m = tf.transpose(m)
  return m

@tf.function
def skew_matrix(tx, ty):
  '''t is angle of skew in degrees on a -1 to 1 coordinate system'''
  tx = tf.math.tan(tf.cast(tx, tf.float32))
  ty = tf.math.tan(tf.cast(ty, tf.float32))
  m = tf.stack([1, tx, 0, ty, 1, 0, 0, 0, 1])
  m = tf.reshape(m, [3,3])
  m = tf.transpose(m)
  return m

@tf.function
def projection_matrix(r):
  '''r is a 1D scalar with 4 values'''
  m = tf.stack([1, 0, r[0], 0, 1, r[1], r[2], r[3], 1])
  m = tf.reshape(m, [3,3])
  m = tf.transpose(m)
  return m

@tf.function
def scale_matrix(r):
  '''r is a 1D scalar with 4 values'''
  m = tf.stack([1*r, 0, 0, 0, 1*r, 0, 0, 0, 1])
  m = tf.reshape(m, [3,3])
  m = tf.transpose(m)
  return m

@tf.function
def apply_transformation_matrix(values):
  '''Can be used as a lambda function'''
  image = values[0]
  T = values[1]

  if tf.rank(image) == 3: image = image[None, ...]
  batch = tf.shape(image)[0]
  height = tf.shape(image)[1]
  width = tf.shape(image)[2]
  channel = tf.shape(image)[3]

  x,y = tf.meshgrid(tf.linspace(-1.0, 1.0, height), tf.linspace(-1.0, 1.0, width))
  coord = tf.stack([x,y, tf.ones_like(x)], axis=-1) # create (xi, yi, 1) vector
  coord = tf.ones([batch,1,1,1])*coord[None, ...]
  T = tf.ones([batch, 1, 1])*T[None,...]

  coord = tf.reshape(coord, [batch, -1, 3])
  G = tf.matmul(coord, T)
  G = tf.reshape(G, [batch, height, width, 3])
  G = G[..., 0:2]/G[..., 2][..., None]
  G = ((G+1)/2)*tf.cast(tf.stack([height, width]) - 1, tf.float32)[None, None, None, :]
  resampled_image = tfa.image.resampler(image, G)
  return resampled_image

@tf.function
def equalize_histogram(image, bins=65535):
    # Compute the histogram of an image, returns as a float from 0 to 1.
    image_dtype = image.dtype
    image = (image - tf.reduce_min(image))/(tf.reduce_max(image) - tf.reduce_min(image))
    image = tf.round(image*bins)
    image = tf.cast(image, tf.int32)

    histo = tf.histogram_fixed_width(image, [0, bins], nbins=bins+1)
    # For the purposes of computing the step, filter out the nonzeros.
    nonzero_histo = tf.boolean_mask(histo, histo != 0)
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // bins
    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    if step == 0: result = image
    else:
        lut_values = (tf.cumsum(histo, exclusive=True) + (step // 2)) // step
        lut_values = tf.clip_by_value(lut_values, 0, bins)
        result = tf.gather(lut_values, image)
    result = tf.cast(result, tf.float32) 
    result = (result - tf.reduce_min(result))/(tf.reduce_max(result) - tf.reduce_min(result))
    return result
