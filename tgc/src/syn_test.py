from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from HGQ.hls4ml_hook import convert_from_hgq_model


def syn_test(model, weight_path, conf, X, Y, N=None):

    save_path = Path(conf.save_path)
    model.load_weights(weight_path)

    hls_prj_path = save_path / 'hls4ml_prj'

    print('Converting to hls4ml model...')
    model_hls = convert_from_hgq_model(
        model,
        hls_config=None,
        output_dir=str(hls_prj_path),
        project_name='TGCNN',
        part='xcvu13p-fhga2104-2l-e',
        clock_period=6.25,
        io_type='io_parallel',
        bias_accum=conf.syn.bias_accum,
        inline_everything=False,
    )

    print('Compiling hls4ml model...')
    model_hls.compile()

    print('Preparing data...')
    N = N or X[0].shape[0]

    X = [
        np.ascontiguousarray(X[0][:N]).astype(np.float32),
        np.ascontiguousarray(X[1][:N]).astype(np.float32),
        np.ascontiguousarray(X[2][:N]).astype(np.float32),
    ]
    Y = np.ascontiguousarray(Y[:N]).astype(np.float32)

    print(f'Running inference on {N} samples...')
    pred_keras = model.predict(X, verbose=0, batch_size=2048)
    diff = pred_keras.ravel() - Y.ravel()
    std_keras = np.sqrt(np.mean(diff**2))
    std_cutoff_keras = np.sqrt(np.mean((diff[abs(diff) < 30])**2))
    print(f'Keras std: {std_keras:.2f} ({std_cutoff_keras:.2f})')

    pred_hls = np.array(model_hls.predict(X))
    diff = np.array(pred_hls).ravel() - Y.ravel()
    std_hls = np.sqrt(np.mean(diff**2))
    std_cutoff_hls = np.sqrt(np.mean((diff[abs(diff) < 30])**2))
    print(f'HLS4ML std: {std_hls:.2f} ({std_cutoff_hls:.2f})')

    ndiff = np.sum(np.any(pred_hls - pred_keras != 0, axis=1))
    print(f'{ndiff} / {N} predictions are different')
    with open(save_path / 'ndiff.txt', 'w') as f:
        f.write(f'{ndiff} / {N} predictions are different\n')
        f.write(f'keras std: {std_keras} ({std_cutoff_keras})\n')
        f.write(f'hls4ml std: {std_hls} ({std_cutoff_hls})')

    if ndiff / N < 0.1:
        return

    plt.close('all')
    plt.hist(pred_keras.ravel() - pred_hls.ravel(), bins=100, range=(-10, 10))
    plt.xlabel('Keras - HLS4ML')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.savefig(save_path / 'diff.pdf', bbox_inches='tight')
