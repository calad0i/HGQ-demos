from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from nn_utils import PBarCallback, SaveTopN, plot_history, compute_bops, save_history, load_history


def test(model, weight_path, save_path: Path, Xt, Xv, X, Y):
    
    save_path = Path(save_path)
    weight_path = weight_path or save_path / 'last.h5'
    model.load_weights(weight_path)
    history = load_history(save_path / 'history.pkl.zst')

    fig, ax = plot_history(history, metrics=('loss', 'val_loss'), ylabel='Loss')
    plt.savefig(save_path / 'loss.pdf')
    
    fig, ax = plot_history(history, ('accuracy', 'val_accuracy'), ylabel='Accuracy')
    plt.savefig(save_path / 'accuracy.pdf', dpi=300)

    fig, ax = plot_history(history, ('multi',), ylabel='BOPs')
    plt.savefig(save_path / 'bops.pdf', dpi=300)

    _ = compute_bops(model, Xt, bsz=2048, verbose=False)
    mul_bops = compute_bops(model, Xv, bsz=2048, rst=False)

    pred = model.predict(X, batch_size=2048, verbose=0)
    acc = np.mean(np.argmax(pred, axis=1) == Y.numpy().ravel())

    print(f'Test accuracy: {acc} @ {mul_bops:.0f} BOPs')
    with open(save_path / 'test_acc.txt', 'w') as f:
        f.write(f'test_accuracy: {acc}\n')
        f.write(f'mul_bops: {mul_bops}')
