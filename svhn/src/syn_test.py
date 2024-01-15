from pathlib import Path
import numpy as np
from HGQ.proxy import to_proxy_model

from hls4ml.converters import convert_from_keras_model


def syn_test(model, save_path, X, Y, N=None):

    save_path = Path(save_path)

    hls_prj_path = save_path / 'hls4ml_prj'
    proxy = to_proxy_model(model, unary_lut_max_table_size=1024)
    print('Converting to hls4ml model...')
    model_hls = convert_from_keras_model(
        proxy,
        hls_config=None,
        output_dir=str(hls_prj_path),
        project_name='svhn',
        part='xcvu9p-flga2104-2L-e',
        clock_period=5,
        io_type='io_stream',
    )

    print('Compiling hls4ml model...')
    model_hls.compile()

    print('Preparing data...')
    X = np.ascontiguousarray(X).astype(np.float32)
    Y = np.ascontiguousarray(Y)

    N = N or X.shape[0]
    print(f'Running inference on {N} samples...')
    pred_keras = model.predict(X[:N], verbose=0, batch_size=2048)
    pred_hls = model_hls.predict(X[:N])

    keras_acc = np.mean(np.argmax(pred_keras, axis=1) == Y[:N])
    hls_acc = np.mean(np.argmax(pred_hls, axis=1) == Y[:N])
    print(f'Keras accuracy: {keras_acc:.5%}')
    print(f'hls4ml accuracy: {hls_acc:.5%}')

    ndiff = np.sum(np.any(pred_hls - pred_keras != 0, axis=1))
    print(f'{ndiff} / {N} predictions are different')
    with open(save_path / 'ndiff.txt', 'w') as f:
        f.write(f'{ndiff} / {N} predictions are different')
        f.write(f'keras accuracy: {keras_acc}')
        f.write(f'hls4ml accuracy: {hls_acc}')
