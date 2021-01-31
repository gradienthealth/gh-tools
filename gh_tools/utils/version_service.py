import tensorflow as tf
import os
from gh_tools import callbacks
import tempfile
from pathlib import Path
import parse
import glob
from datetime import datetime

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
        self.callbacks = self.init_callbacks()

    def init_callbacks(self):
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

        return [self.tensorboard, 
                self.log_code,
                self.saving, 
                self.saving_weights, 
                self.saving_h5,
                self.keras_storage]

    def get_best(self, monitor=None, mode=None, save_format=None):
        monitor = monitor if monitor is not None else self.monitor
        mode = mode if mode is not None else self.mode
        save_format = save_format if save_format is not None else self.save_format

        files = glob.glob(os.path.join(
          self.keras_storage.local_dir, 
          'models',
          '*.h5'
        ))

        parselist = [parse.parse(save_format, os.path.basename(os.path.splitext(f)[0]))[monitor] for f in files]
        if mode is 'max':
          index = parselist.index(max(parselist))
        else:
          index = parselist.index(min(parselist))
        return os.path.splitext(files[index])[0]

    def load_model(self, path=None, version=None, compile=False):
        if version is not None:
          path = os.path.join(
            self.keras_storage.local_dir, 
            'models', 
            version
          )
        elif path is None:
          path = self.get_best()
        return tf.keras.models.load_model(path, compile=compile), os.path.basename(path)

    def save_model(self, model, tag='manual', name=None):
        tag = tag + '-' + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        name = '' if name is None else '-' + name

        model.save(os.path.join(self.keras_storage.local_dir, 'manual_models', tag + name))
        model.save_weights(os.path.join(self.keras_storage.local_dir, 'manual_models', tag + name))
        model.save(os.path.join(self.keras_storage.local_dir, 'manual_models', tag + name + '.h5'))
        self.keras_storage.sync()
