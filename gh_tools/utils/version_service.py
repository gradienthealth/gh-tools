import tensorflow as tf
import os
from gh_tools import callbacks
import tempfile
from pathlib import Path

VERSION = '0.1.0'

class VersionService:
    def __init__(self, 
      namespace, 
      version=None, 
      endpoint_url='https://s3.wasabisys.com',
      save_format='model.{epoch:02d}-{val_loss:.10f}',
      monitor='val_loss',
      verbose=1,
      save_freq='epoch',
      mode='min',
      code_dir = '.'):
        self.namespace = namespace
        self.version = version
        self.endpoint_url = endpoint_url
        self.save_format = save_format
        self.monitor = monitor
        self.verbose = verbose
        self.save_freq = save_freq
        self.mode = mode
        self.code_dir = code_dir

    def callbacks(self):
        self.keras_storage = callbacks.KerasStorage(
          self.namespace, 
          version=self.version, 
          endpoint_url=self.endpoint_url)
        
        version_tmp_file = '/tmp/{}.version'.format(VERSION)
        Path(version_tmp_file).touch()
        tf.io.gfile.copy(
          version_tmp_file, 
          os.path.join(self.keras_storage.job_dir, 
          '{}.version'.format(VERSION)
        ), overwrite=True)

        self.log_code = callbacks.LogCode(
          self.keras_storage.job_dir, 
          self.code_dir, 
          endpoint_url=self.endpoint_url)
          
        self.tensorboard = tf.keras.callbacks.TensorBoard(
          log_dir=self.keras_storage.job_dir + '/tensorboard', 
          write_graph=True, 
          update_freq=self.save_freq)

        self.saving = tf.keras.callbacks.ModelCheckpoint(
          os.path.join(self.keras_storage.local_dir, 'models', self.save_format), 
          monitor=self.monitor, 
          verbose=self.verbose, 
          save_freq=self.save_freq, 
          mode=self.mode)

        self.saving_weights = tf.keras.callbacks.ModelCheckpoint(
          os.path.join(self.keras_storage.local_dir, 'models', self.save_format), 
          save_weights_only=True,
          monitor=self.monitor, 
          verbose=self.verbose, 
          save_freq=self.save_freq, 
          mode=self.mode)

        self.saving_h5 = tf.keras.callbacks.ModelCheckpoint(
          os.path.join(self.keras_storage.local_dir, 'models', self.save_format + '.h5'),
          monitor=self.monitor, 
          verbose=self.verbose, 
          save_freq=self.save_freq, 
          mode=self.mode)

        return [
                self.tensorboard, 
                self.log_code,
                self.saving, 
                self.saving_weights, 
                self.saving_h5,
                self.keras_storage]
