
from pathlib import Path

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from nn_utils import PBarCallback, SaveTopN, plot_history, trace_minmax, save_history, load_history, absorb_batchNorm, get_best_ckpt

from HGQ.bops import trace_minmax, FreeBOPs, ResetMinMax



def train_fp32(model, X, Y, Xv, Yv, save_path, cdr_args: dict, bsz: int, epochs: int, acc_thres: float):

    print('Compiling model & registering callbacks...')
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    opt = tf.keras.optimizers.Adam(1, amsgrad=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    _cdr = tf.keras.experimental.CosineDecayRestarts(**cdr_args)

    def cdr(epoch):
        thres = epochs - 10
        if epoch <= thres:
            return _cdr(epoch)
        return cdr(thres) * (0.9 ** (epoch - thres))
    scheduler = tf.keras.callbacks.LearningRateScheduler(cdr)

    pbar = PBarCallback(metric='loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.2%}/{val_accuracy:.2%} - lr:{lr:.2e}')
    save = SaveTopN(
        metric_fn=lambda x: x['val_accuracy'],  # (x['val_accuracy']-0.85)/x['multi'],
        n=20,
        path=save_path / 'ckpts',
        cond_fn=lambda x: x['val_accuracy'] > acc_thres,
        fname_format='epoch={epoch}-acc={accuracy:.2%}-val_acc={val_accuracy:.2%}.h5'
    )

    callbacks = [scheduler, pbar, save]

    batch_size = bsz

    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(Xv, Yv), callbacks=callbacks, verbose=0)  # type: ignore
    history = model.history.history  # type: ignore
    save_history(history, save_path / 'history.pkl.zst')
    model.save_weights(save_path / 'last.h5')

    fig, ax = plot_history(history, metrics=('loss', 'val_loss'), ylabel='Loss')
    plt.savefig(save_path / 'loss.pdf')
    fig, ax = plot_history(history, metrics=('accuracy', 'val_accuracy'), ylabel='Accuracy')
    plt.savefig(save_path / 'accuracy.pdf')

    return model, history


def train_hgq(model, model_fp32, X, Y, Xv, Yv, Xs, Ys, save_path:Path, cdr_args:dict, bsz:int, epochs:int, acc_thres: float):

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    pred = model_fp32.predict(Xs, batch_size=2048, verbose=0)
    acc_fp32 = np.mean(np.argmax(pred, axis=1) == Ys.numpy().ravel())
    print(f'FP32 accuracy: {acc_fp32:.2%}')
    
    absorb_batchNorm(model, model_fp32)
    
    pred = model.predict(Xs, batch_size=2048, verbose=0)
    hgq_acc_1 = np.mean(np.argmax(pred, axis=1) == Ys.numpy().ravel())
    print(f'pre-training HGQ accuracy: {hgq_acc_1:.2%}')
    
    with open(save_path / 'pretrain_acc.txt', 'w') as f:
        f.write(f'FP32 accuracy: {acc_fp32:.2%}\n')
        f.write(f'pre-training HGQ accuracy: {hgq_acc_1:.2%}\n')
    
    print('Compiling model & registering callbacks...')
    opt = tf.keras.optimizers.Adam(1, amsgrad=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    model.compile(optimizer=opt, loss=loss, metrics=metrics)


    _cdr = tf.keras.experimental.CosineDecayRestarts(**cdr_args)
    def cdr(epoch):
        thres = epochs - 10
        if epoch <= thres:
            return _cdr(epoch)
        return cdr(thres) * (0.9 ** (epoch - thres))
    scheduler = tf.keras.callbacks.LearningRateScheduler(cdr)
    
    pbar = PBarCallback(metric='loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.2%}/{val_accuracy:.2%} - lr:{lr:.2e}')
    
    save = SaveTopN(
        metric_fn=lambda x: (x['val_accuracy']-0.85)/x['multi'],
        n=20,
        path=save_path / 'ckpts',
        cond_fn=lambda x: x['val_accuracy'] > acc_thres and x['multi'] < 4e4,
        fname_format='epoch={epoch}-acc={accuracy:.2%}-val_acc={val_accuracy:.2%}-metric={metric:.2e}-BOPs={multi:.2e}.h5'
    )

    rst = ResetMinMax()
    bops = FreeBOPs()
    
    callbacks = [scheduler, bops, pbar, save, rst]
    
    batch_size = bsz
    
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(Xv, Yv), callbacks=callbacks, verbose=0) # type: ignore
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
    