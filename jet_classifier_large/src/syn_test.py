import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# from HGQ.hls4ml_hook import convert_from_hgq_model
from HGQ import shutup
from HGQ.proxy import to_proxy_model
from hls4ml.converters import convert_from_keras_model
from HGQ import trace_minmax
from HGQ.proxy.fixed_point_quantizer import FixedPointQuantizer
from HGQ.proxy.unary_lut import UnaryLUT
import keras

# from hls4ml.optimization.distributed_arithmetic.resoure_surrogate import ResourceSurrogate
# from hls4ml.optimization.distributed_arithmetic import compiler_config


def syn_test(save_path: Path, X, Y, N=None, softmax=False):

    with open(save_path / 'test_acc.json') as f:
        results = json.load(f)

    X = np.ascontiguousarray(X).astype(np.float32)
    Y = np.ascontiguousarray(Y)
    N = N or X.shape[0]
    print(f'Running inference on {N} samples')
    pbar = tqdm(list(save_path.glob('proxy_models/*.h5')))
    co = {'FixedPointQuantizer': FixedPointQuantizer, 'UnaryLUT': UnaryLUT}
    (save_path / 'hls4ml_prjs').mkdir(exist_ok=True, parents=True)

    # compiler_config(backend='da4ml', enabled=True)
    for ckpt in pbar:

        model: keras.Model = keras.models.load_model(ckpt, custom_objects=co)  # type: ignore

        hls_prj_path = save_path / f'hls4ml_prjs/{ckpt.stem.split("-")[0]}'

        with shutup:
            # latency strategy for this model won't meet II=1 requirement in general
            model_hls = convert_from_keras_model(
                model,
                hls_config={'Model': {'Precision': 'ap_fixed<1,0>', 'ReuseFactor': 1, 'Strategy': 'distributed_arithmetic'}},
                output_dir=str(hls_prj_path),
                project_name='jet_classifier_large',
                part='xcvu9p-flga2104-2L-e',
                clock_period=5,
                io_type='io_parallel',
                backend='vitis',
            )
            model_hls.write()
            model_hls._compile()

        pred_keras = model.predict(X[:N], verbose=0, batch_size=16384)  # type: ignore
        pred_hls = model_hls.predict(X[:N])
        hls_acc = np.mean(np.argmax(pred_hls, axis=1) == Y.ravel())
        print(f'HLS accuracy: {hls_acc:.5%}')
        ndiff = int(np.sum(np.any(pred_hls - pred_keras != 0, axis=1)))
        results[ckpt.name]['hls_acc'] = hls_acc
        results[ckpt.name]['ndiff'] = ndiff

        if ndiff > 0:
            print(f'{ndiff} out of {N} samples differ for {ckpt.name}')
        with open(save_path / 'test_acc.json', 'w') as f:
            json.dump(results, f)
