import os
import pickle as pkl
from pathlib import Path

import zstd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


def get_data(data_path: Path, n_constituents, mmap_location='/cpu:0', seed=42):

    import h5py as h5
    with h5.File(data_path / '150c-train.h5') as f:
        X_train_val = np.array(f['feature'][:16, :n_constituents])  # type: ignore
        y_train_val = np.array(f['label'])
    with h5.File(data_path / '150c-test.h5') as f:
        X_test = np.array(f['feature'][:16, :n_constituents])  # type: ignore
        y_test = np.array(f['label'])
    labels = 'gqWZt'

    # scaler = StandardScaler()
    # X_train_val = scaler.fit_transform(X_train_val)
    # X_test = scaler.transform(X_test)

    with tf.device(mmap_location):  # type: ignore
        X_train_val = tf.convert_to_tensor(X_train_val, dtype=tf.float16)
        y_train_val = tf.convert_to_tensor(y_train_val, dtype=tf.float16)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float16)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float16)

    return X_train_val, X_test, y_train_val, y_test
