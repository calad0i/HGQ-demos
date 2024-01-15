import tensorflow as tf
import h5py as h5
import numpy as np


def get_mask(fname, thres=0.01):
    with h5.File(fname, 'r') as f:
        mask12 = np.array(f['corr/12']) > thres
        mask13 = np.array(f['corr/13']) > thres
        mask23 = np.array(f['corr/23']) > thres
    return mask12, mask13, mask23


def get_data_and_mask(fname, seed=42, split=(0.7, 0.1, 0.2), mask_thres=0.01, mmap_location='/cpu:0'):
    with h5.File(fname, "r") as f:
        X, Y = np.array(f["X"]).transpose(0, 2, 1), np.array(f["Y"])
    rng = np.random.default_rng(seed)
    split = np.cumsum(split)
    split = (split * len(X) / split[-1]).astype(np.int32)
    order = np.arange(len(X))
    rng.shuffle(order)
    X, Y = X[order], Y[order]
    X_train, X_val, X_test = X[:split[0]], X[split[0]:split[1]], X[split[1]:]
    y_train, y_val, y_test = Y[:split[0]], Y[split[0]:split[1]], Y[split[1]:]

    try:
        mask12, mask13, mask23 = get_mask(fname, thres=mask_thres)
        print("Loaded masks from cache")
    except:
        print("Failed to load masks, generating and caching...")
        m1 = np.any(X_train[:, :, :3], axis=2)
        m2 = np.any(X_train[:, :, 3:5], axis=2)
        m3 = np.any(X_train[:, :, 5:7], axis=2)
        m12 = np.concatenate([m1, m2], axis=1)
        m13 = np.concatenate([m1, m3], axis=1)
        m23 = np.concatenate([m2, m3], axis=1)
        corr12 = np.corrcoef(m12.T)
        corr13 = np.corrcoef(m13.T)
        corr23 = np.corrcoef(m23.T)
        with h5.File(fname, 'r+') as f:
            f.create_dataset('corr/12', data=corr12)
            f.create_dataset('corr/13', data=corr13)
            f.create_dataset('corr/23', data=corr23)
        mask12 = corr12 > mask_thres
        mask13 = corr13 > mask_thres
        mask23 = corr23 > mask_thres

    with tf.device(mmap_location):  # type: ignore
        X_train = [
            tf.convert_to_tensor(X_train[:, :, :3]),
            tf.convert_to_tensor(X_train[:, :, 3:5]),
            tf.convert_to_tensor(X_train[:, :, 5:])
        ]
        X_val = [
            tf.convert_to_tensor(X_val[:, :, :3]),
            tf.convert_to_tensor(X_val[:, :, 3:5]),
            tf.convert_to_tensor(X_val[:, :, 5:])
        ]
        X_test = [
            tf.convert_to_tensor(X_test[:, :, :3]),
            tf.convert_to_tensor(X_test[:, :, 3:5]),
            tf.convert_to_tensor(X_test[:, :, 5:])
        ]
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    mask12 = tf.convert_to_tensor(mask12[:50, 50:], dtype=tf.float32)
    mask13 = tf.convert_to_tensor(mask13[:50, 50:], dtype=tf.float32)
    mask23 = tf.convert_to_tensor(mask23[:50, 50:], dtype=tf.float32)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (mask12, mask13, mask23)

# def get_mask
