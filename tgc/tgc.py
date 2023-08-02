
import sys
sys.path.append('../')


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import mplhep as hep
hep.style.use(hep.style.CMS)

from pathlib import Path

from pathlib import Path

from nn_utils import set_seed, get_best_ckpt
from src.dataloader import get_data_and_mask
from src.model import get_model_fp32, get_model_fhq
from src.train import train_fp32, train_fhq
from src.test import test
from src.syn_test import syn_test

from FHQ.bops import compute_bops

import omegaconf
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--run', '-r', nargs='*', type=str, default=['all'])
    parser.add_argument('--ckpt_fp32', '-kf', type=str, default=None)
    parser.add_argument('--ckpt_fhq', '-kq', type=str, default=None)
    args = parser.parse_args()

    conf = omegaconf.OmegaConf.load(args.config)
    cfp32 = conf.fp32
    cfhq = conf.fhq

    print('Setting seed...')
    set_seed(conf.seed)

    print('Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (mask12, mask13, mask23) = get_data_and_mask(conf.data_path, seed=conf.seed, split=conf.splits, mask_thres=conf.mask_thres)

    print('Creating models...')
    if conf.fp32.model.masking:
        model_fp32 = get_model_fp32(mask12, mask13, mask23)
    else:
        model_fp32 = get_model_fp32(None, None, None)
    
    if conf.fhq.model.masking:
        model_fhq = get_model_fhq(mask12, mask13, mask23, conf=conf.fhq.model)
    else:
        model_fhq = get_model_fhq(None, None, None, conf=conf.fhq.model)

    if 'all' in args.run or 'train_fp32' in args.run:
        print('Phase: train_fp32')
        _ = train_fp32(model_fp32, X_train, y_train, X_val, y_val, conf.fp32)

    if 'all' in args.run or 'train_fhq' in args.run:
        ckpt_fp32 = args.ckpt_fp32 or get_best_ckpt(Path(conf.fp32.save_path) / 'ckpts', take_min=True)
        print(f'Using checkpoint: {ckpt_fp32}')
        model_fp32.load_weights(ckpt_fp32)

        print('Phase: train_fhq')
        _ = train_fhq(model_fhq, model_fp32, X_train, y_train, X_val, y_val, X_test, y_test, conf.fhq)

    ckpt_fhq = args.ckpt_fhq or get_best_ckpt(Path(conf.fhq.save_path) / 'ckpts', take_min=True)
    bops_computed = False

    if 'all' in args.run or 'test' in args.run:
        print('Phase: test')
        print(f'Using checkpoint: {ckpt_fhq}')
        test(model_fhq, ckpt_fhq, conf.fhq.save_path, X_train, X_val, X_test, y_test)
        bops_computed = True

    if 'all' in args.run or 'syn' in args.run:
        print('Phase: syn_test')
        print(f'Using checkpoint: {ckpt_fhq}')
        if not bops_computed:
            model_fhq.load_weights(ckpt_fhq)
            compute_bops(model_fhq, X_train, bsz=16384, verbose=False)
            bops = compute_bops(model_fhq, X_val, bsz=16384, rst=False)
            print(f'BOPS: {bops}')

        syn_test(model_fhq, ckpt_fhq, conf.fhq, X_test, y_test, None)
