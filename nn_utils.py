

from io import BytesIO
import json
import os
import random

from glob import glob
from pathlib import Path
import pickle as pkl
from typing import Callable

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import zstd
import h5py as h5

from keras.saving.legacy import hdf5_format
from keras.layers.convolutional.base_conv import Conv
from keras.layers import Dense

from HGQ.bops import compute_bops


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):  # type: ignore
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class SaveTopN(keras.callbacks.Callback):
    def __init__(self,
                 metric_fn: Callable[[dict], float],
                 n: int,
                 path: str | Path,
                 side: str = 'max',
                 fname_format='epoch={epoch}-metric={metric:.4e}.h5',
                 cond_fn: Callable[[dict], bool] = lambda x: True,
                 ):
        self.n = n
        self.metric_fn = metric_fn
        self.path = Path(path)
        self.fname_format = fname_format
        os.makedirs(path, exist_ok=True)
        self.weight_paths = np.full(n, '/dev/null', dtype=object)
        if side == 'max':
            self.best = np.full(n, -np.inf)
            self.side = np.greater
        elif side == 'min':
            self.best = np.full(n, np.inf)
            self.side = np.less
        self.cond = cond_fn

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(logs, dict)
        assert isinstance(self.model, keras.models.Model)
        logs = logs.copy()
        logs['epoch'] = epoch
        if not self.cond(logs):
            return
        metric = self.metric_fn(logs)

        if self.side(metric, self.best[-1]):
            try:
                os.remove(self.weight_paths[-1])
            except OSError:
                pass
            logs['metric'] = metric
            fname = self.path / self.fname_format.format(**logs)
            self.best[-1] = metric
            self.weight_paths[-1] = fname
            self.model.save_weights(fname)
            with h5.File(fname, 'r+') as f:
                log_str = json.dumps(logs, cls=NumpyFloatValuesEncoder)
                f.attrs['train_log'] = log_str
            idx = np.argsort(self.best)
            if self.side == np.greater:
                idx = idx[::-1]
            self.best = self.best[idx]
            self.weight_paths = self.weight_paths[idx]

    def rename_ckpts(self, dataset, bsz=65536):
        assert self.weight_paths[0] != '/dev/null', 'No checkpoints to rename'
        assert isinstance(self.model, keras.models.Model)

        weight_buf = BytesIO()
        with h5.File(weight_buf, 'w') as f:
            hdf5_format.save_weights_to_hdf5_group(f, self.model)
        weight_buf.seek(0)

        for i, path in enumerate(tqdm(self.weight_paths, desc='Renaming checkpoints')):
            if path == '/dev/null':
                continue
            self.model.load_weights(path)
            bops = compute_bops(self.model, dataset, bsz=bsz, verbose=False)
            with h5.File(path, 'r+') as f:
                logs = json.loads(f.attrs['train_log'])  # type: ignore
                logs['multi'] = bops
                metric = self.metric_fn(logs)
                logs['metric'] = metric
                f.attrs['train_log'] = json.dumps(logs, cls=NumpyFloatValuesEncoder)
            self.best[i] = metric
            new_fname = self.path / self.fname_format.format(**logs)
            os.rename(path, new_fname)
            self.weight_paths[i] = new_fname

        idx = np.argsort(self.best)
        self.best = self.best[idx]
        self.weight_paths = self.weight_paths[idx]
        with h5.File(weight_buf, 'r') as f:
            hdf5_format.load_weights_from_hdf5_group_by_name(f, self.model)


class PBarCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric='loss: {loss:.2f}/{val_loss:.2f}'):
        self.pbar = None
        self.template = metric

    def on_epoch_begin(self, epoch, logs=None):
        if self.pbar is None:
            self.pbar = tqdm(total=self.params['epochs'], unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.pbar, tqdm)
        assert isinstance(logs, dict)
        self.pbar.update(1)
        string = self.template.format(**logs)
        if 'multi' in logs:
            string += f' - BOPs: {logs["multi"]:,.0f}'
        self.pbar.set_description(string)

    def on_train_end(self, logs=None):
        if self.pbar is not None:
            self.pbar.close()


def plot_history(histry: dict, metrics=('loss', 'val_loss'), ylabel='Loss', logy=False):
    fig, ax = plt.subplots()
    for metric in metrics:
        ax.plot(histry[metric], label=metric)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.legend()
    return fig, ax


def save_model(model: keras.models.Model, path: str):
    _path = Path(path)
    model.save(path)
    if model.history is not None:
        history = model.history.history
    else:
        history = {}
    with open(_path.with_suffix('.history'), 'wb') as f:
        f.write(zstd.compress(pkl.dumps(history)))


def load_model(path: str, co=None):
    _path = Path(path)
    model: keras.Model = keras.models.load_model(path, custom_objects=co)  # type: ignore
    with open(_path.with_suffix('.history'), 'rb') as f:
        history: dict[str, list] = pkl.loads(zstd.decompress(f.read()))
    return model, history


def save_history(history, path):
    with open(path, 'wb') as f:
        f.write(zstd.compress(pkl.dumps(history)))


def load_history(path):
    with open(path, 'rb') as f:
        history = pkl.loads(zstd.decompress(f.read()))
    return history


def absorb_batchNorm(model_target, model_original):
    for layer in model_target.layers:
        if layer.__class__.__name__ == 'Functional':
            absorb_batchNorm(layer, model_original.get_layer(layer.name))
            continue
        if (isinstance(layer, Dense) or isinstance(layer, Conv)) and \
            len(nodes := model_original.get_layer(layer.name)._outbound_nodes) > 0 and \
                isinstance(nodes[0].outbound_layer, keras.layers.BatchNormalization):
            _gamma, _beta, _mu, _var = model_original.get_layer(layer.name)._outbound_nodes[0].outbound_layer.get_weights()
            _ratio = _gamma / np.sqrt(0.001 + _var)
            _bias = -_gamma * _mu / np.sqrt(0.001 + _var) + _beta

            k, *_b = model_original.get_layer(layer.name).get_weights()
            if _b:
                b = _b[0]
            else:
                b = np.zeros(layer.output_shape[-1])
            nk = np.einsum('...c, c-> ...c', k, _ratio, optimize=True)
            nb = np.einsum('...c, c-> ...c', b, _ratio, optimize=True) + _bias
            extras = layer.get_weights()[2:]
            layer.set_weights([nk, nb, *extras])
        elif hasattr(layer, 'kernel'):
            for w in layer.weights:
                if '_bw' not in w.name:
                    break
            else:
                continue
            weights = layer.get_weights()
            new_weights = model_original.get_layer(layer.name).get_weights()
            l = len(new_weights)
            layer.set_weights([*new_weights, *weights[l:]][:len(weights)])


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    tf.config.experimental.enable_op_determinism()


import h5py as h5
import json


def get_best_ckpt(save_path: Path, take_min=False):
    ckpts = list(save_path.glob('*.h5'))

    def rank(ckpt: Path):
        with h5.File(ckpt, 'r') as f:
            log: dict = f.attrs['train_log']  # type: ignore
        log = json.loads(log)  # type: ignore
        metric = log['metric']  # type: ignore
        return metric

    ckpts = sorted(ckpts, key=rank, reverse=not take_min)
    ckpt = ckpts[0]
    return ckpt
