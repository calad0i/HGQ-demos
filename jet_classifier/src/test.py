from pathlib import Path
import numpy as np


from matplotlib import pyplot as plt
from nn_utils import plot_history, trace_minmax, load_history


def test(model, weight_path, save_path: Path, Xt, X, Y):
    weight_path = weight_path or save_path / 'last.h5'
    model.load_weights(weight_path)
    history = load_history(save_path / 'history.pkl.zst')

    fig, ax = plot_history(history, ('accuracy', 'val_accuracy'))
    ax.set_ylim(0.7, 0.77)
    plt.savefig(save_path / 'accuracy.pdf', dpi=300)

    fig, ax = plot_history(history, ('bops',), ylabel='BOPs')
    ax.set_ylim(np.min(history['bops']) * 0.7, np.min(history['bops']) * 2)
    plt.savefig(save_path / 'bops.pdf', dpi=300)

    mul_bops = trace_minmax(model, Xt, bsz=664000)

    pred = model.predict(X, batch_size=16384, verbose=0)  # type: ignore
    acc = np.mean(np.argmax(pred, axis=1) == Y.ravel())
    './train_history/ckpts/epoch=3389-acc=75.20%-val_acc=75.22%-BOPs=1824.0-metric=1.2176e-05.h5'
    print(f'Test accuracy: {acc:.5%} @ {mul_bops} BOPs')
    with open(save_path / 'test_acc.txt', 'w') as f:
        f.write(f'test_accuracy: {acc}\n')
        f.write(f'mul_bops: {mul_bops}')
