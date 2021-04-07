import os
import tensorflow as tf
from datetime import datetime
import subprocess

def download_and_load_model(path, endpoint_url='https://s3.wasabisys.com'):
    file_extension = os.path.basename(path).split('.')[-1]
    tmp_dir = '/tmp/' + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

    if file_extension == 'h5' or file_extension == 'h5py':
        command = "aws s3 cp '{}' '{}/{}' --endpoint-url {}".format(path, tmp_dir, os.path.basename(path), endpoint_url)
        subprocess.run(command, shell=True, env=os.environ.copy(), check=True)
        return tf.keras.models.load_model(os.path.join(tmp_dir, os.path.basename(path)))
    else:
        command = "aws s3 sync '{}' '{}' --endpoint-url {} --exact-timestamps".format(path, tmp_dir, endpoint_url)
        subprocess.run(command, shell=True, env=os.environ.copy(), check=True)
        return tf.keras.models.load_model(tmp_dir, compile=False)
