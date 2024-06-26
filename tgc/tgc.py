
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
from src.model import get_model_hgq
from src.train import train_hgq
from src.test import test
from src.syn_test import syn_test

from HGQ.bops import trace_minmax

import omegaconf
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--run', '-r', nargs='*', type=str, default=['all'])
    parser.add_argument('--ckpt', '-kq', type=str, default=None)
    args = parser.parse_args()

    conf = omegaconf.OmegaConf.load(args.config)

    print('Setting seed...')
    set_seed(conf.seed)

    print('Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (mask12, mask13, mask23) = get_data_and_mask(conf.data_path, seed=conf.seed, split=conf.splits, mask_thres=conf.mask_thres)

    print('Creating models...')
    if conf.model.masking:
        model_hgq = get_model_hgq(mask12, mask13, mask23, conf=conf.model)
    else:
        model_hgq = get_model_hgq(None, None, None, conf=conf.model)

    if 'all' in args.run or 'train' in args.run:
        print('Phase: train_hgq')
        _ = train_hgq(model_hgq, X_train, y_train, X_val, y_val, X_test, y_test, conf)

    if 'all' in args.run or 'test' in args.run:
        print('Phase: test')
        test(model_hgq, conf.save_path, X_train, X_val, X_test, y_test)
        bops_computed = True

    if 'syn' in args.run or 'all' in args.run:
        print('Phase: syn')
        syn_test(Path(conf.save_path), X_test, y_test, N=None)
