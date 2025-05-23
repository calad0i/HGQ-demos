from pathlib import Path
import json

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from HGQ import to_proxy_model, shutup
from hls4ml.converters import convert_from_keras_model
from HGQ.proxy.fixed_point_quantizer import FixedPointQuantizer
from HGQ.proxy.unary_lut import UnaryLUT
import keras


from .model import Diag


def syn_test(save_path: Path, X, Y, N=None, softmax=False):

    N = N or X[0].shape[0]
    X = [
        np.ascontiguousarray(X[0][:N]).astype(np.float32),
        np.ascontiguousarray(X[1][:N]).astype(np.float32),
        np.ascontiguousarray(X[2][:N]).astype(np.float32),
    ]
    Y = np.ascontiguousarray(Y[:N]).astype(np.float32)

    print(f'Running inference on {N} samples')
    pbar = tqdm(list(save_path.glob('proxy_models/*.h5')))
    co = {'FixedPointQuantizer': FixedPointQuantizer, 'UnaryLUT': UnaryLUT, 'Diag': Diag}
    (save_path / 'hls4ml_prjs').mkdir(exist_ok=True, parents=True)

    with open(save_path / 'test_acc.json', 'r') as f:
        results = json.load(f)

    for ckpt in pbar:

        model: keras.Model = keras.models.load_model(ckpt, custom_objects=co)  # type: ignore

        hls_prj_path = save_path / f'hls4ml_prjs/{ckpt.stem.split("-")[0]}'
        hls_config = {
            # replace latency by distributed_arithmetic to enable DA
            'Model': {'Precision': 'ap_fixed<-1,0>', 'ReuseFactor': 1, 'Strategy': 'latency'},
            # These three layers should always use latency, as the inputs are 1-bit and DA won't help
            'LayerName': {
                'm1_conv': {'Strategy': 'latency'},
                'm2_conv': {'Strategy': 'latency'},
                'm3_conv': {'Strategy': 'latency'},
            }
        }

        with shutup:
            model_hls = convert_from_keras_model(
                model,
                hls_config=hls_config,
                output_dir=str(hls_prj_path),
                project_name='TGCNN',
                part='xcvu13p-fhga2104-2l-e',
                clock_period=6.25,
                io_type='io_parallel',
                backend='vitis'
            )
            model_hls.compile()

        pred_keras = model.predict(X, verbose=0, batch_size=8192)  # type: ignore
        pred_hls = model_hls.predict(X)

        diff = pred_hls != pred_keras

        if np.any(diff):
            pbar.write(f'{np.sum(diff)} out of {N} samples differ for {ckpt.name}. Samples: {pred_keras[diff][:5]}, {pred_hls[diff][:5]}')

        diff = pred_hls.ravel() - Y.ravel()  # type: ignore
        std_hls = np.sqrt(np.mean(diff**2))
        std_cutoff_hls = np.sqrt(np.mean((diff[abs(diff) < 30])**2))

        cost = sum(l.attributes['da_kernel_cost'] for l in model_hls.graph.values() if 'da_kernel_cost' in l.attributes)
        results[ckpt.name]['hls_std'] = float(std_hls)
        results[ckpt.name]['hls_std_cutoff'] = float(std_cutoff_hls)
        results[ckpt.name]['hls_cost'] = int(cost)
        with open(save_path / 'test_acc.json', 'w') as f:
            json.dump(results, f)
