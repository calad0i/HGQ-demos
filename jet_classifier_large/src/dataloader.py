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
        X_train_val = np.array(f['feature'])  # type: ignore
        y_train_val = np.array(f['label'])
    with h5.File(data_path / '150c-test.h5') as f:
        X_test = np.array(f['feature'])  # type: ignore
        y_test = np.array(f['label'])
    labels = 'gqWZt'

    X_train_val = X_train_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    scale = np.std(X_train_val, axis=(0, 1))
    bias = np.mean(X_train_val, axis=(0, 1))

    X_train_val = (X_train_val[:, :n_constituents] - bias) / scale
    X_test = (X_test[:, :n_constituents] - bias) / scale

    # X_train_val = X_train_val[..., [5, 8, 11]]
    # X_test = X_test[..., [5, 8, 11]]

    with tf.device(mmap_location):  # type: ignore
        X_train_val = tf.convert_to_tensor(X_train_val, dtype=tf.float16)
        y_train_val = tf.convert_to_tensor(y_train_val, dtype=tf.float16)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float16)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float16)

    return X_train_val, X_test, y_train_val, y_test
