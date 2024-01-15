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
from src.dataloader import get_data
from src.model import get_model
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
    parser.add_argument('--ckpt', '-k', type=str, default=None)
    args = parser.parse_args()

    conf = omegaconf.OmegaConf.load(args.config)

    print('Setting seed...')
    set_seed(conf.seed)

    data_path = conf.datapath

    print('Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(data_path, val_split=conf.val_split, seed=conf.seed, mmap_location='/gpu:0')

    print('Creating models...')

    model_hgq = get_model(conf)

    set_seed(conf.seed)
    if 'all' in args.run or 'train' in args.run:

        print('Phase: train_hgq')
        _ = train_hgq(model_hgq,
                      X_train,
                      y_train,
                      X_val,
                      y_val,
                      X_test,
                      y_test,
                      conf,
                      )

    if 'all' in args.run or 'test' in args.run:
        print('Phase: test')
        test(model_hgq, Path(conf.save_path), X_train, X_val, X_test, y_test)
        bops_computed = True

    # if 'all' in args.run or 'syn' in args.run:
    #     print('Phase: syn')
    #     print(f'Using checkpoint: {ckpt_hgq}')
    #     if not bops_computed:
    #         print('Computing BOPS...')
    #         model_hgq.load_weights(ckpt_hgq)
    #         _ = trace_minmax(model_hgq, X_train, bsz=2048, verbose=False)
    #         bops = trace_minmax(model_hgq, X_val, bsz=2048, rst=False)
    #         print(f'BOPS: {bops}')
    #     syn_test(model_hgq, ckpt_hgq, conf.save_path, X_test, y_test)
