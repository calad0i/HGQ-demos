
from pathlib import Path

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from nn_utils import PBarCallback, save_history
from nn_utils import BetaScheduler, PeratoFront

from HGQ.bops import FreeBOPs, ResetMinMax
from omegaconf import OmegaConf

def train_hgq(model, X, Y, Xs, Ys, conf):

    save_path = Path(conf.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'config.yaml', 'w') as f:
        OmegaConf.save(conf, f)

    pred = model.predict(Xs, batch_size=2048, verbose=0)
    hgq_acc_1 = np.mean(np.argmax(pred, axis=1) == Ys.numpy().ravel())
    print(f'pre-training HGQ accuracy: {hgq_acc_1:.2%}')

    with open(save_path / 'pretrain_acc.txt', 'w') as f:
        f.write(f'pre-training HGQ accuracy: {hgq_acc_1:.2%}\n')

    print('Compiling model & registering callbacks...')
    opt = tf.keras.optimizers.Adam(1, amsgrad=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    assert conf.train.cdr_args['t_mul'] == 1
    alpha_len = conf.train.cdr_args.pop('alpha_steps')
    _cdr = tf.keras.experimental.CosineDecayRestarts(**conf.train.cdr_args)
    cycle_len = conf.train.cdr_args['first_decay_steps']
    cos_len = cycle_len - alpha_len

    def cdr(epoch):
        n_cycle = epoch // cycle_len
        cur_cycle = epoch % cycle_len
        if cur_cycle < cos_len:
            return _cdr(cur_cycle * cycle_len / cos_len + n_cycle * cycle_len)
        return _cdr.alpha

    scheduler = tf.keras.callbacks.LearningRateScheduler(cdr)

    pbar = PBarCallback(metric='loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.2%}/{val_accuracy:.2%} - lr:{lr:.2e} - beta: {beta:.2e}')

    save = PeratoFront(
        path=save_path / 'ckpts',
        cond_fn=lambda x: True,
        fname_format='epoch={epoch}-acc={accuracy:.2%}-val_acc={val_accuracy:.2%}-BOPs={bops}.h5',
        metrics_names=['val_accuracy', 'bops'],
        sides=[1, -1],
    )

    rst = ResetMinMax()
    bops = FreeBOPs()
    beta_sched = BetaScheduler.from_config(conf)

    callbacks = [scheduler, beta_sched, bops, pbar, save, rst]

    batch_size = conf.train.bsz

    model.fit(X, Y, batch_size=batch_size, epochs=conf.train.epochs, validation_split=0.1, callbacks=callbacks, verbose=0)  # type: ignore
    history = model.history.history  # type: ignore

    save_history(history, save_path / 'history.pkl.zst')
    model.save_weights(save_path / 'last.h5')

    print('Recomputing BOPs...')
    save.rename_ckpts(X, bsz=2048)
    return model, history
