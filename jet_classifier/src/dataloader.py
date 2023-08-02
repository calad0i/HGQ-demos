import os
import pickle as pkl
from pathlib import Path

import zstd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

def get_data(data_path: Path, mmap_location = '/cpu:0', seed=42):
    if not os.path.exists(data_path):
        print('Downloading data...')
        data = fetch_openml('hls4ml_lhc_jets_hlf')
        buf = pkl.dumps(data)
        with open(data_path, 'wb') as f:
            f.write(zstd.compress(buf))
    else:
        with open(data_path, 'rb') as f:
            data = pkl.loads(zstd.decompress(f.read()))

    X, y = data['data'], data['target']
    codecs = {'g': 0, 'q': 1, 't': 4, 'w': 2, 'z': 3}
    y = np.array([codecs[i] for i in y])

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    X_train_val, X_test, y_train_val, y_test = X_train_val.astype(np.float32), X_test.astype(np.float32), y_train_val, y_test

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)
    
    with tf.device(mmap_location): # type: ignore
        X_train_val = tf.convert_to_tensor(X_train_val, dtype=tf.float32)
        y_train_val = tf.convert_to_tensor(y_train_val, dtype=tf.float32)

    return X_train_val, X_test, y_train_val, y_test
