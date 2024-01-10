
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from nn_utils import plot_history, load_history
from HGQ.bops import trace_minmax


def test(model, weight_path, save_path: Path, Xt, Xv, X, Y):

    save_path = Path(save_path)
    weight_path = weight_path or save_path / 'last.h5'
    model.load_weights(weight_path)
    history = load_history(save_path / 'history.pkl.zst')

    fig, ax = plot_history(history, metrics=('loss', 'val_loss'), ylabel='Loss')
    plt.savefig(save_path / 'loss.pdf', bbox_inches='tight')

    fig, ax = plot_history(history, ('mse', 'val_mse'), ylabel='MSE [degree$^2$]')
    ax.set_ylim(2.5, 6)
    plt.savefig(save_path / 'mse.pdf', dpi=300, bbox_inches='tight')

    fig, ax = plot_history(history, ('bops',), ylabel='BOPs')
    ax.set_yscale('log')
    plt.savefig(save_path / 'bops.pdf', dpi=300, bbox_inches='tight')

    _ = trace_minmax(model, Xt, bsz=16384, verbose=False)
    mul_bops = trace_minmax(model, Xv, bsz=16384, rst=False)

    pred = model.predict(X, batch_size=16384, verbose=0)  # type: ignore
    diff = pred.ravel() - Y.numpy().ravel()
    std_hgq = np.sqrt(np.mean(diff**2))
    std_cutoff_hgq = np.sqrt(np.mean((diff[abs(diff) < 30])**2))
    print(f'Test std: {std_hgq:.2f} ({std_cutoff_hgq:.2f}) @ {mul_bops:.0f} BOPs')

    with open(save_path / 'test_std.txt', 'w') as f:
        f.write(f'test_std: {std_hgq} ({std_cutoff_hgq})\n')
        f.write(f'mul_bops: {mul_bops}\n')
