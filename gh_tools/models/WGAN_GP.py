import tensorflow as tf

# Define WGAN - GP
class WGAN_GP(tf.keras.Model):
  def __init__(self, discriminator, generator):
    super(WGAN_GP, self).__init__()
    self.discriminator = discriminator
    self.generator = generator
    self.n_critic = tf.Variable(2, dtype=tf.int64)

  @tf.function
  def wasserstein_loss(self, y_true, y_pred):
    '''
      y_true is of shape (BATCH, 1) and labels the validity of the sample.
      This line expands y_true with empty dimensions to match y_pred.
      For example, patchgan would have (BATCH, SIZE_X, SIZE_Y, 1) as y_pred.
      But WGAN typically will have (BATCH, 1).
    ''' 
    # We flip the y_true label to make it easy to train on supervised tasks
    y_true = -1*tf.reshape(y_true, tf.concat([[tf.shape(y_true)[0]], 
                                               tf.ones(tf.rank(y_pred) - 1, tf.int32)], axis=0))
    return tf.reduce_mean(y_true * y_pred)

  @tf.function
  def gradient_penalty_loss(self, grads):
    gradients_sqr = tf.square(grads)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=tf.range(1, tf.rank(grads)))
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.square(1 - gradient_l2_norm)
    return tf.reduce_mean(gradient_penalty)

  def compile(self, 
              d_optimizer, 
              g_optimizer, 
              g_loss_fn=None, 
              d_loss_fn=None, 
              n_critic = 5, 
              loss_weights = [1, 1, 10]):
    super(WGAN_GP, self).compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.optimizers = [d_optimizer, g_optimizer]
    self.d_loss_fn = self.wasserstein_loss if d_loss_fn is None else d_loss_fn
    self.g_loss_fn = tf.keras.losses.MeanAbsoluteError() if g_loss_fn is None else g_loss_fn
    self.loss_weights = loss_weights # [gen, disc, gp]

    self.n_critic.assign(n_critic)

    self.g_loss_disc_tracker = tf.keras.metrics.Mean(name="g_loss_disc")
    self.g_loss_gen_tracker = tf.keras.metrics.Mean(name="g_loss_gen")
    self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

    self.d_loss_disc_tracker = tf.keras.metrics.Mean(name="d_loss_disc")
    self.gp_loss_tracker = tf.keras.metrics.Mean(name="gp_loss")
    self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")

  @property
  def metrics(self):
    return [self.g_loss_disc_tracker, 
            self.g_loss_gen_tracker,
            self.g_loss_tracker,
            self.d_loss_disc_tracker,
            self.gp_loss_tracker,
            self.d_loss_tracker]

  @tf.function(experimental_relax_shapes=True)
  def forward_discriminator(self, x, y):
    batch_size = tf.shape(y)[0]
    labels = tf.concat([tf.ones((batch_size, 1), tf.float32),
                        -1*tf.ones((batch_size, 1), tf.float32)], axis=0)
    
    predictions = self.discriminator(tf.concat([self.generator(x), y], axis=0))
    return self.d_loss_fn(labels, predictions)

  @tf.function(experimental_relax_shapes=True)
  def forward_generator(self, x, y):
    batch_size = tf.shape(y)[0]
    misleading_labels = -1*tf.ones((batch_size, 1), tf.float32)
    outputs = self.generator(x)
    predictions = self.discriminator(outputs)
    g_loss_disc = self.d_loss_fn(misleading_labels, predictions)
    g_loss_gen  = self.g_loss_fn(y, outputs)
    return g_loss_gen, g_loss_disc

  @tf.function(experimental_relax_shapes=True)
  def forward_gradient_penalty(self, x, y):
    batch_size = tf.shape(y)[0]

    fake_images = self.generator(x)
    alpha = tf.random.uniform((batch_size, 1, 1, 1))
    interpolated_img = (alpha * y) + ((1 - alpha) * fake_images)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(interpolated_img)
      pred = self.discriminator(interpolated_img)
    grads = tape.gradient(pred, interpolated_img)
    gp_loss = self.gradient_penalty_loss(grads)
    return gp_loss

  def call(self, inputs):
    generated_images = self.generator(inputs)
    predictions = self.discriminator(generated_images)
    return generated_images, predictions

  @tf.function(experimental_relax_shapes=True)
  def train_step(self, data):
    lambda_gen = self.loss_weights[0]
    lambda_disc = self.loss_weights[1]
    lambda_gp = self.loss_weights[2]
    conditional_images = data[0]
    real_images = data[1]

    # train discriminator
    if lambda_disc > 0:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self.discriminator.trainable_weights)
        d_loss_disc = self.forward_discriminator(conditional_images, real_images)
        gp_loss = self.forward_gradient_penalty(conditional_images, real_images)
        d_loss = d_loss_disc*lambda_disc + gp_loss*lambda_gp
      self.gp_loss_tracker.update_state(gp_loss)
      self.d_loss_disc_tracker.update_state(d_loss_disc)
      self.d_loss_tracker.update_state(d_loss)
      grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
      self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

    if lambda_gen > 0:
      # train generator
      if self.d_optimizer.iterations % self.n_critic == 0:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
          tape.watch(self.generator.trainable_weights)
          g_loss_gen, g_loss_disc = self.forward_generator(conditional_images, real_images)
          g_loss = g_loss_gen*lambda_gen + g_loss_disc*lambda_disc
        self.g_loss_disc_tracker.update_state(g_loss_disc)
        self.g_loss_gen_tracker.update_state(g_loss_gen)
        self.g_loss_tracker.update_state(g_loss)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

    return {m.name: m.result() for m in self.metrics}

  @tf.function
  def test_step(self, data):
    lambda_gen = self.loss_weights[0]
    lambda_disc = self.loss_weights[1]
    conditional_images = data[0]
    real_images = data[1]

    d_loss_disc = self.forward_discriminator(conditional_images, real_images)
    g_loss_gen, g_loss_disc = self.forward_generator(conditional_images, real_images)
    self.d_loss_disc_tracker.update_state(d_loss_disc)
    self.g_loss_disc_tracker.update_state(g_loss_disc)
    self.g_loss_gen_tracker.update_state(g_loss_gen)
    return {m.name: m.result() for m in self.metrics}
