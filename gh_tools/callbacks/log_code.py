import os
import subprocess
import time
from datetime import datetime
import tensorflow as tf
from gh_tools.utils import colab

class LogCode(tf.keras.callbacks.Callback):
    def __init__(self, job_dir, code_dir='.', endpoint_url = 'https://s3.wasabisys.com'):
        super().__init__()
        self.code_dir = code_dir
        self.job_dir = job_dir
        self.endpoint_url = endpoint_url
        self.started = False
    
    def save_and_upload_notebook(self, version):
        notebook_filename = os.path.basename(os.environ["NOTEBOOK_FILEPATH"])
        print('saving notebook...')
        colab.save_notebook()
        tf.io.gfile.copy(os.environ["NOTEBOOK_FILEPATH"], '{}/code/{}/{}'.format(
          self.job_dir, 
          version, 
          notebook_filename))
        print('saving complete.')

    def upload(self):
        datetime_string = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        if colab.isColabNotebook(): self.save_and_upload_notebook(datetime_string)
        command = "aws s3 sync '{}' '{}/code/{}' --endpoint-url {} --exact-timestamps --exclude '*.pyc' --exclude '*.git/*' --exclude '*.ipynb_checkpoints*'".format(
          self.code_dir, 
          self.job_dir, 
          datetime_string, 
          self.endpoint_url)
        subprocess.run(command, shell=True, env=os.environ.copy())
        
    def on_epoch_end(self, *args, **kwargs):
        if not self.started:
          x = time.time()
          print('Uploading code to:', self.job_dir)
          self.upload()
          print('Completed code upload in {0:10.2f}s'.format(time.time() - x))
          self.started = True

    def on_train_end(self, logs=None):
        self.started = False
  