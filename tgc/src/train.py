from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from nn_utils import PBarCallback, SaveTopN, save_history, plot_history, absorb_batchNorm

from FHQ.bops import ResetMinMax, FreeBOPs


def train_fp32(model: keras.Model, X, Y, Xv, Yv, conf):
    cdr = keras.experimental.CosineDecayRestarts(**conf.train.cdr_args)
    scheduler = keras.callbacks.LearningRateScheduler(cdr)
    save_path = Path(conf.save_path)

    loss = 'mse'
    opt = 'adam'
    pbar = PBarCallback('loss: {loss:.2f}/{val_loss:.2f} - mse: {mse:.2f}/{val_mse:.2f} - lr: {lr:.2e}')
    save = SaveTopN(
        metric_fn=lambda x: x['val_mse'],
        n=10,
        path=save_path / 'ckpts',
        side='min',
        cond_fn=lambda x: x['val_mse'] < conf.train.mse_thres,
        fname_format='epoch={epoch}-mse={mse:.3f}-val_mse={val_mse:.3f}.h5'
    )

    model.compile(loss=loss, optimizer=opt, metrics=['mse'], jit_compile=True)
    history = model.fit(X, Y, validation_data=(Xv, Yv), epochs=conf.train.epochs, callbacks=[scheduler, pbar, save], batch_size=conf.train.bsz, verbose=0)  # type: ignore
    history = history.history

    save_history(history, save_path / 'history.pkl.zst')
    model.save_weights(save_path / 'last.h5')

    fig, ax = plot_history(history, metrics=('loss', 'val_loss'), ylabel='Loss')
    plt.savefig(save_path / 'loss.pdf', bbox_inches='tight')
    fig, ax = plot_history(history, metrics=('mse', 'val_mse'), ylabel='MSE')
    ax.set_ylim(2.5, 5)
    plt.savefig(save_path / 'mse.pdf', bbox_inches='tight')

    return model, history


def train_fhq(model: keras.Model, model_fp32, X, Y, Xv, Yv, Xs, Ys, conf):

    absorb_batchNorm(model, model_fp32)

    pred = model_fp32.predict(Xs, batch_size=16384, verbose=0)  # type: ignore
    diff = pred.ravel() - Ys.numpy().ravel()
    std_fp32 = np.sqrt(np.mean(diff**2))
    std_cutoff_fp32 = np.sqrt(np.mean((diff[abs(diff) < 30])**2))
    print(f'FP32 std: {std_fp32:.2f} ({std_cutoff_fp32:.2f})')

    pred = model.predict(Xs, batch_size=16384, verbose=0)  # type: ignore
    diff = pred.ravel() - Ys.numpy().ravel()
    std_fhq = np.sqrt(np.mean(diff**2))
    std_cutoff_fhq = np.sqrt(np.mean((diff[abs(diff) < 30])**2))
    print(f'pre-training FHQ std: {std_fhq:.2f} ({std_cutoff_fhq:.2f})')

    save_path = Path(conf.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    with open(save_path / 'std.txt', 'w') as f:
        f.write(f'FP32 std: {std_fhq:.2f} ({std_cutoff_fhq:.2f})\n')
        f.write(f'pre-training FHQ std: {std_fhq:.2f} ({std_cutoff_fhq:.2f})\n')

    cdr = keras.experimental.CosineDecayRestarts(**conf.train.cdr_args)
    scheduler = keras.callbacks.LearningRateScheduler(cdr)

    loss = 'mse'
    opt = 'adam'
    pbar = PBarCallback('loss: {loss:.2f}/{val_loss:.2f} - mse: {mse:.2f}/{val_mse:.2f} - lr: {lr:.2e}')
    save = SaveTopN(
        metric_fn=lambda x: (x['val_mse'] - 2) * x['multi'],
        n=10,
        path=save_path / 'ckpts',
        side='min',
        cond_fn=lambda x: x['val_mse'] < conf.train.mse_thres,
        fname_format='epoch={epoch}-mse={mse:.3}-val_mse={val_mse:.3}-metric={metric:.2e}-BOPs={multi:.0f}.h5'
    )
    rst = ResetMinMax()
    bops = FreeBOPs()

    model.compile(loss=loss, optimizer=opt, metrics=['mse'])
    history = model.fit(X, Y, validation_data=(Xv, Yv), epochs=conf.train.epochs, callbacks=[bops, scheduler, pbar, save, rst], batch_size=conf.train.bsz, verbose=0)  # type: ignore

    history = history.history

    save_history(history, save_path / 'history.pkl.zst')
    model.save_weights(save_path / 'last.h5')

    print('Recomputing BOPs...')
    save.rename_ckpts(X, bsz=16384)

    return model, history
