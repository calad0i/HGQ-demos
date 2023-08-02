import tensorflow as tf
from pathlib import Path

from FHQ.bops import FreeBOPs, ResetMinMax
from nn_utils import PBarCallback, SaveTopN, save_history


from FHQ import set_default_kernel_quantizer_config
def train(model, X, Y, save_path: Path, lr: float, epochs: int, bsz: int, val_split: float, acc_thres: float):

    print('Compiling model...')
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    print('Registering callbacks...')
    bops = FreeBOPs()

    pbar = PBarCallback(metric='loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.2%}/{val_accuracy:.2%}')
    rst = ResetMinMax()
    save = SaveTopN(
        metric_fn=lambda x: (min(x['val_accuracy'],x['accuracy']) - 0.71) / x['multi'],
        n=20,
        path=save_path / 'ckpts',
        cond_fn=lambda x: min(x['val_accuracy'],x['accuracy']) > acc_thres and x['multi'] < 10000,
        fname_format='epoch={epoch}-acc={accuracy:.2%}-val_acc={val_accuracy:.2%}-BOPs={multi}-metric={metric:.4e}.h5'
    )

    callbacks = [bops, pbar, save, rst]

    print('Start training...')
    model.fit(X, Y, epochs=epochs, batch_size=bsz, validation_split=val_split, verbose=0, callbacks=callbacks)  # type: ignore
    history: dict[str, list] = model.history.history  # type: ignore
    save_history(history, save_path / 'history.pkl.zst')
    model.save_weights(save_path / 'last.h5')
    save.rename_ckpts(X)

    return model, history
