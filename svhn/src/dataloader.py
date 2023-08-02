import gc
import os
from pathlib import Path

import numpy as np
import scipy.io

import tensorflow as tf


def get_data(data_path, val_split=0.2, seed=42, mmap_location='/cpu:0'):
    path = Path(data_path)
    train = scipy.io.loadmat(path / 'train_32x32.mat')  # type: ignore
    test = scipy.io.loadmat(path / 'test_32x32.mat')  # type: ignore
    extra = scipy.io.loadmat(path / 'extra_32x32.mat')  # type: ignore

    lt = train['y'].shape[0]
    loc = int(lt * (1 - val_split))
    rng = np.random.default_rng(seed=seed)
    idx = rng.permutation(lt)
    train['X'] = train['X'][:, :, :, idx]
    train['y'] = train['y'][idx]
    raw_X_train, raw_y_train = train['X'][:, :, :, :loc], train['y'][:loc]
    raw_X_val, raw_y_val = train['X'][:, :, :, loc:], train['y'][loc:]
    raw_X_test, raw_y_test = test['X'], test['y']
    raw_X_extra, raw_y_extra = extra['X'], extra['y']

    del train, test, extra

    X_train = np.empty((raw_X_train.shape[3] + raw_X_extra.shape[3], 32, 32, 3), dtype=np.float32)

    X_val = np.empty((raw_X_val.shape[3], 32, 32, 3), dtype=np.float32)

    X_test = np.empty((raw_X_test.shape[3], 32, 32, 3), dtype=np.float32)

    X_train[:raw_X_train.shape[3]] = np.transpose(raw_X_train, (3, 0, 1, 2))
    X_train[raw_X_train.shape[3]:] = np.transpose(raw_X_extra, (3, 0, 1, 2))
    del raw_X_train, raw_X_extra
    gc.collect()

    X_val[:] = np.transpose(raw_X_val, (3, 0, 1, 2))
    del raw_X_val

    X_test[:] = np.transpose(raw_X_test, (3, 0, 1, 2))
    del raw_X_test

    X_train /= 255.
    X_val /= 255.
    X_test /= 255.

    y_train = np.concatenate((raw_y_train, raw_y_extra), axis=0).ravel()
    y_val = raw_y_val.ravel()
    y_test = raw_y_test.ravel()

    y_train %= 10
    y_val %= 10
    y_test %= 10


    with tf.device(mmap_location): # type: ignore
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float16)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float16)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.int8)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float16)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.int8)
        
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
