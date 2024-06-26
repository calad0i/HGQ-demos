
from pathlib import Path
import json

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from nn_utils import plot_history, load_history
from HGQ.bops import trace_minmax
from HGQ import to_proxy_model


def test(model, save_path: Path | str, Xt, Xv, X, Y):

    save_path = Path(save_path)

    history = load_history(save_path / 'history.pkl.zst')

    fig, ax = plot_history(history, metrics=('loss', 'val_loss'), ylabel='Loss')
    plt.savefig(save_path / 'loss.pdf', bbox_inches='tight')

    fig, ax = plot_history(history, ('mse', 'val_mse'), ylabel='MSE [degree$^2$]')
    ax.set_ylim(2.5, 6)
    plt.savefig(save_path / 'mse.pdf', dpi=300, bbox_inches='tight')

    fig, ax = plot_history(history, ('bops',), ylabel='BOPs')
    ax.set_yscale('log')
    plt.savefig(save_path / 'bops.pdf', dpi=300, bbox_inches='tight')

    (save_path / 'proxy_models').mkdir(exist_ok=True, parents=True)
    results = {}
    pbar = tqdm(list(save_path.glob('ckpts/*.h5')))
    for ckpt in pbar:
        model.load_weights(ckpt)
        _ = trace_minmax(model, Xt, bsz=2048, verbose=False)
        bops = trace_minmax(model, Xv, bsz=2048, rst=False, verbose=False)
        proxy = to_proxy_model(model, unary_lut_max_table_size=1024)
        proxy.save(save_path / f'proxy_models/{ckpt.stem}.h5')

        pred = model.predict(X, batch_size=16384, verbose=0)
        proxy_pred = proxy.predict(X, batch_size=16384, verbose=0)
        diff = pred.ravel() - Y.numpy().ravel()
        std_keras = np.sqrt(np.mean(diff**2))
        std_cutoff_keras = np.sqrt(np.mean((diff[abs(diff) < 30])**2))
        diff = proxy_pred.ravel() - Y.numpy().ravel()
        std_proxy = np.sqrt(np.mean(diff**2))
        std_cutoff_proxy = np.sqrt(np.mean((diff[abs(diff) < 30])**2))

        mismatch = pred - proxy_pred
        n_mismatch = np.sum(np.any(mismatch != 0, axis=1))
        if n_mismatch > 0:
            pbar.write(f'{n_mismatch} out of {len(X[0])} samples differ for {ckpt.name}; HLS error: {std_keras:.2f}/{std_cutoff_keras:.2f}')
            pbar.write(f'Sample: {pred[mismatch!=0][:5]}, {proxy_pred[mismatch!=0][:5]}')

        results[ckpt.name] = {'std': float(std_keras), 'std_cutoff': float(std_cutoff_keras), 'bops': int(bops), 'hls_std': float(std_proxy), 'hls_std_cutoff': float(std_cutoff_proxy)}
        pbar.set_description(f'Test accuracy: {std_keras:.2f}/{std_cutoff_keras:.2f} @ {bops:.0f} BOPs')

    with open(save_path / 'test_acc.json', 'w') as f:
        json.dump(results, f)
