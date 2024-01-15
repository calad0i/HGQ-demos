from pathlib import Path
import json

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from nn_utils import PBarCallback, SaveTopN, plot_history, trace_minmax, save_history, load_history


def test(model, save_path: Path, Xt, Xv, X, Y):

    save_path = Path(save_path)
    history = load_history(save_path / 'history.pkl.zst')

    fig, ax = plot_history(history, metrics=('loss', 'val_loss'), ylabel='Loss')
    plt.savefig(save_path / 'loss.pdf')

    fig, ax = plot_history(history, ('accuracy', 'val_accuracy'), ylabel='Accuracy')
    plt.savefig(save_path / 'accuracy.pdf', dpi=300)

    fig, ax = plot_history(history, ('bops',), ylabel='BOPs')
    plt.savefig(save_path / 'bops.pdf', dpi=300)

    ckpts = save_path.glob('ckpts/*.h5')

    results = {}
    pbar = tqdm(list(save_path.glob('ckpts/*.h5')))
    for ckpt in pbar:
        model.load_weights(ckpt)
        _ = trace_minmax(model, Xt, bsz=2048, verbose=False)
        bops = trace_minmax(model, Xv, bsz=2048, rst=False, verbose=False)

        pred = model.predict(X, batch_size=2048, verbose=0)
        acc = np.mean(np.argmax(pred, axis=1) == Y.numpy().ravel())

        print(f'Test accuracy: {acc} @ {bops:.0f} BOPs')
        results[ckpt.name] = {'acc': acc, 'bops': bops}
        pbar.set_description(f'Test accuracy: {acc:.5%} @ {bops:.0f} BOPs')

    with open(save_path / 'test_acc.json', 'w') as f:
        json.dump(results, f)
