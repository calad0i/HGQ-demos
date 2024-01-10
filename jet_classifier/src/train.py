import tensorflow as tf
from pathlib import Path
import keras

from HGQ.bops import FreeBOPs, ResetMinMax, CalibratedBOPs
from nn_utils import PBarCallback, save_history, PeratoFront, BetaScheduler


def train(model:keras.Model, X, Y, save_path: Path, conf):

    print('Compiling model...')
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    opt = tf.keras.optimizers.Adam(learning_rate=conf.train.lr)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    print('Registering callbacks...')
    bops = FreeBOPs() if not conf.train.calibrated_bops else CalibratedBOPs(X[:conf.train.calibrated_bops])
    cos = tf.keras.experimental.CosineDecay(conf.train.lr, conf.train.epochs)
    sched = tf.keras.callbacks.LearningRateScheduler(cos)
    pbar = PBarCallback(metric='loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.2%}/{val_accuracy:.2%} - lr: {lr:.2e} - beta: {beta:.2e}')
    rst = ResetMinMax()

    save = PeratoFront(
        path=save_path / 'ckpts',
        cond_fn=lambda x: True,
        fname_format='epoch={epoch}-acc={accuracy:.2%}-val_acc={val_accuracy:.2%}-BOPs={bops}.h5',
        metrics_names=['val_accuracy', 'bops'],
        sides=[1, -1],
    )
    beta_sched = BetaScheduler.from_config(conf)
    callbacks = [sched, beta_sched, bops, pbar, save, rst]

    print('Start training...')
    model.fit(X, Y,
              epochs=conf.train.epochs,
              batch_size=conf.train.bsz,
              validation_split=conf.train.val_split,
              verbose=0, # type: ignore
              callbacks=callbacks
              )
    history: dict[str, list] = model.history.history  # type: ignore
    save_history(history, save_path / 'history.pkl.zst')
    model.save_weights(save_path / 'last.h5')
    save.rename_ckpts(X)

    return model, history
