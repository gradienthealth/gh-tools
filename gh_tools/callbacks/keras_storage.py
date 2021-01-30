import os
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import subprocess

class KerasStorage(tf.keras.callbacks.Callback):
    def __init__(self, namespace, version = None, endpoint_url = 'https://s3.wasabisys.com'):
        super()
        self.version = datetime.now().strftime("%m-%d-%Y-%H:%M:%S") if version is None else version
        self.local_dir = '/tmp/{}/{}'.format(namespace, self.version)
        self.job_dir = 's3://gradienthealth-logs/{}/{}'.format(namespace, self.version)
        self.endpoint_url = endpoint_url
        Path(self.local_dir).mkdir(parents=True, exist_ok=True)
        self.sync()
        
    def sync(self):
        command = "aws s3 sync '{}' '{}' --endpoint-url {} --exact-timestamps".format(self.local_dir, self.job_dir, self.endpoint_url)
        subprocess.run(command, shell=True, env=os.environ.copy())
        command = "aws s3 sync '{}' '{}' --endpoint-url {} --exact-timestamps".format(self.job_dir, self.local_dir, self.endpoint_url)
        subprocess.run(command, shell=True, env=os.environ.copy())

    def on_epoch_end(self, epoch, logs):
        self.sync()
