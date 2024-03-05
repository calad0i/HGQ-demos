from pathlib import Path
import json

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from HGQ import to_proxy_model

from nn_utils import PBarCallback, SaveTopN, plot_history, trace_minmax, save_history, load_history


def test(model, save_path: Path, Xt, Xv, X, Y):
    history = load_history(save_path / 'history.pkl.zst')

    fig, ax = plot_history(history, ('accuracy', 'val_accuracy'))
    plt.savefig(save_path / 'accuracy.pdf', dpi=300)

    fig, ax = plot_history(history, ('bops',), ylabel='BOPs')
    ax.set_ylim(np.min(history['bops']) * 0.7, np.min(history['bops']) * 2)
    plt.savefig(save_path / 'bops.pdf', dpi=300)

    (save_path / 'proxy_models').mkdir(exist_ok=True, parents=True)
    results = {}
    Y = np.array(Y)
    pbar = tqdm(list(save_path.glob('ckpts/*.h5')))
    for ckpt in pbar:
        model.load_weights(ckpt)
        _ = trace_minmax(model, Xt, bsz=16384, verbose=False)
        bops = trace_minmax(model, Xv, bsz=16384, verbose=False, rst=False)
        proxy = to_proxy_model(model, unary_lut_max_table_size=-1)
        proxy.save(save_path / f'proxy_models/{ckpt.stem}.h5')

        pred = model.predict(X, batch_size=16384, verbose=0)
        proxy_pred = proxy.predict(X, batch_size=16384, verbose=0)  # type: ignore

        acc = np.mean(np.argmax(pred, axis=1) == Y.ravel())
        hls_acc = np.mean(np.argmax(proxy_pred, axis=1) == Y.ravel())
        mismatch = pred - proxy_pred
        n_mismatch = np.sum(np.any(mismatch != 0, axis=1))
        if n_mismatch > 0:
            print(f'{n_mismatch} out of {X.shape[0]} samples differ for {ckpt.name}; HLS accuracy: {hls_acc:.5%}')
            print(f'Sample: {pred[mismatch!=0][:5]}, {proxy_pred[mismatch!=0][:5]}')

        # print(f'Test accuracy: {acc:.5%} @ {mul_bops} BOPs')
        results[ckpt.name] = {'acc': acc, 'bops': bops, 'hls_acc': hls_acc}
        pbar.set_description(f'Test accuracy: {acc:.5%} @ {bops:.0f} BOPs')
    with open(save_path / 'test_acc.json', 'w') as f:
        json.dump(results, f)
