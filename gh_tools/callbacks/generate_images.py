import tensorflow as tf
import argparse
import numpy as np

class GenerateImages(tf.keras.callbacks.Callback):
    def __init__(self, forward, dataset, log_dir, interval=1000, postfix='val', lambda_fn=None):
        super()
        self.step_count = 0
        self.postfix = postfix
        self.interval = interval
        self.forward = forward
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.dataset_iterator = iter(dataset)
        self.lambda_fn = lambda_fn

    def generate_images(self):
        self.lambda_fn(self)

    def on_batch_begin(self, batch, logs={}):
        self.step_count += 1
        if self.step_count % self.interval == 0:
            self.generate_images()
            
    def on_train_end(self, logs={}):
        self.generate_images()
