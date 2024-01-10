from pathlib import Path

import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

from nn_utils import PBarCallback, SaveTopN, save_history, plot_history, PeratoFront, BetaScheduler

from HGQ.bops import ResetMinMax, FreeBOPs


def train_hgq(model: keras.Model, X, Y, Xv, Yv, Xs, Ys, conf):

    pred = model.predict(Xs, batch_size=16384, verbose=0)  # type: ignore
    diff = pred.ravel() - Ys.numpy().ravel()
    std_hgq = np.sqrt(np.mean(diff**2))
    std_cutoff_hgq = np.sqrt(np.mean((diff[abs(diff) < 30])**2))
    print(f'pre-training HGQ std: {std_hgq:.2f} ({std_cutoff_hgq:.2f})')

    save_path = Path(conf.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    with open(save_path / 'std.txt', 'w') as f:
        f.write(f'FP32 std: {std_hgq:.2f} ({std_cutoff_hgq:.2f})\n')
        f.write(f'pre-training HGQ std: {std_hgq:.2f} ({std_cutoff_hgq:.2f})\n')

    cdr = keras.experimental.CosineDecayRestarts(**conf.train.cdr_args)
    scheduler = keras.callbacks.LearningRateScheduler(cdr)

    loss = 'mse'
    opt = 'adam'
    pbar = PBarCallback('loss: {loss:.2f}/{val_loss:.2f} - mse: {mse:.2f}/{val_mse:.2f} - lr: {lr:.2e} - beta: {beta:.2e}')

    save = PeratoFront(
        path=save_path / 'ckpts',
        cond_fn=lambda x: True,
        fname_format='epoch={epoch}-mse={mse:.3}-val_mse={val_mse:.3}-BOPs={bops:.0f}.h5',
        metrics_names=['val_mse', 'bops'],
        sides=[-1, -1],
    )

    beta_sched = BetaScheduler.from_config(conf)

    rst = ResetMinMax()
    bops = FreeBOPs()

    callbacks = [bops, beta_sched, scheduler, pbar, save, rst]

    model.compile(loss=loss, optimizer=opt, metrics=['mse'])
    history = model.fit(X, Y, validation_data=(Xv, Yv), epochs=conf.train.epochs, callbacks=callbacks, batch_size=conf.train.bsz, verbose=0)  # type: ignore

    history = history.history

    save_history(history, save_path / 'history.pkl.zst')
    model.save_weights(save_path / 'last.h5')

    print('Recomputing BOPs...')
    save.rename_ckpts(X, bsz=16384)
    plot_history(history, save_path / 'history.pdf')

    return model, history
