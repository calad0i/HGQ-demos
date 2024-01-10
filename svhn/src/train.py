
from pathlib import Path

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from nn_utils import PBarCallback, save_history
from nn_utils import BetaScheduler, PeratoFront

from HGQ.bops import FreeBOPs, ResetMinMax


def train_hgq(model, X, Y, Xv, Yv, Xs, Ys, conf):

    save_path = Path(conf.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

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

    _cdr = tf.keras.experimental.CosineDecayRestarts(**conf.train.cdr_args)

    def cdr(epoch):
        thres = conf.train.epochs - 10
        if epoch <= thres:
            return _cdr(epoch)
        return cdr(thres) * (0.9 ** (epoch - thres))
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

    model.fit(X, Y, batch_size=batch_size, epochs=conf.train.epochs, validation_data=(Xv, Yv), callbacks=callbacks, verbose=0)  # type: ignore
    history = model.history.history  # type: ignore

    save_history(history, save_path / 'history.pkl.zst')
    model.save_weights(save_path / 'last.h5')

    # fig, ax = plot_history(history, metrics=('loss', 'val_loss'), ylabel='Loss')
    # plt.savefig(save_path / 'loss.pdf')
    # fig, ax = plot_history(history, metrics=('accuracy', 'val_accuracy'), ylabel='Accuracy')
    # plt.savefig(save_path / 'accuracy.pdf')

    print('Recomputing BOPs...')
    save.rename_ckpts(X, bsz=2048)
    return model, history
