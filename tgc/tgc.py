
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
from src.model import get_model_fp32, get_model_hgq
from src.train import train_fp32, train_hgq
from src.test import test
from src.syn_test import syn_test

from HGQ.bops import trace_minmax

import omegaconf
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--run', '-r', nargs='*', type=str, default=['all'])
    parser.add_argument('--ckpt_fp32', '-kf', type=str, default=None)
    parser.add_argument('--ckpt_hgq', '-kq', type=str, default=None)
    args = parser.parse_args()

    conf = omegaconf.OmegaConf.load(args.config)
    cfp32 = conf.fp32
    chgq = conf.hgq

    print('Setting seed...')
    set_seed(conf.seed)

    print('Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (mask12, mask13, mask23) = get_data_and_mask(conf.data_path, seed=conf.seed, split=conf.splits, mask_thres=conf.mask_thres)

    print('Creating models...')
    if conf.fp32.model.masking:
        model_fp32 = get_model_fp32(mask12, mask13, mask23)
    else:
        model_fp32 = get_model_fp32(None, None, None)

    if conf.hgq.model.masking:
        model_hgq = get_model_hgq(mask12, mask13, mask23, conf=conf.hgq.model)
    else:
        model_hgq = get_model_hgq(None, None, None, conf=conf.hgq.model)

    if 'all' in args.run or 'train_fp32' in args.run:
        print('Phase: train_fp32')
        _ = train_fp32(model_fp32, X_train, y_train, X_val, y_val, conf.fp32)

    if 'all' in args.run or 'train_hgq' in args.run:
        ckpt_fp32 = args.ckpt_fp32 or get_best_ckpt(Path(conf.fp32.save_path) / 'ckpts', take_min=True)
        print(f'Using checkpoint: {ckpt_fp32}')
        model_fp32.load_weights(ckpt_fp32)

        print('Phase: train_hgq')
        _ = train_hgq(model_hgq, model_fp32, X_train, y_train, X_val, y_val, X_test, y_test, conf.hgq)

    ckpt_hgq = args.ckpt_hgq or get_best_ckpt(Path(conf.hgq.save_path) / 'ckpts', take_min=True)
    bops_computed = False

    if 'all' in args.run or 'test' in args.run:
        print('Phase: test')
        print(f'Using checkpoint: {ckpt_hgq}')
        test(model_hgq, ckpt_hgq, conf.hgq.save_path, X_train, X_val, X_test, y_test)
        bops_computed = True

    if 'all' in args.run or 'syn' in args.run:
        print('Phase: syn_test')
        print(f'Using checkpoint: {ckpt_hgq}')
        if not bops_computed:
            model_hgq.load_weights(ckpt_hgq)
            trace_minmax(model_hgq, X_train, bsz=16384, verbose=False)
            bops = trace_minmax(model_hgq, X_val, bsz=16384, rst=False)
            print(f'BOPS: {bops}')

        syn_test(model_hgq, ckpt_hgq, conf.hgq, X_test, y_test, None)
