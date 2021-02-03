import os
import tensorflow as tf
from datetime import datetime
import subprocess

def download_and_load_model(path, endpoint_url='https://s3.wasabisys.com'):
    tmp_dir = '/tmp/' + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    command = "aws s3 sync '{}' '{}' --endpoint-url {} --exact-timestamps".format(path, tmp_dir, endpoint_url)
    subprocess.run(command, shell=True, env=os.environ.copy())
    return tf.keras.models.load_model(tmp_dir, compile=False)
