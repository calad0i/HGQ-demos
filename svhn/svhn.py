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

    data_path = conf.datapath

    print('Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(data_path, val_split=conf.val_split, seed=conf.seed, mmap_location='/gpu:0')

    print('Creating models...')
    model_fp32 = get_model_fp32(renorm=False)

    model_fhq = get_model_fhq(
        init_bw_k=cfhq.model.a_init_bw,
        init_bw_a=cfhq.model.k_init_bw,
        bops_reg_factor=cfhq.model.bops_reg_factor,
        parallel_factors=cfhq.model.parallel_factors,
        l1_cc=cfhq.model.k_bw_l1_reg_conv,
        l1_dc=cfhq.model.k_bw_l1_reg_dense,
        l1_act=cfhq.model.a_bw_l1_reg,
    )

    if 'all' in args.run or 'train_fp32' in args.run:
        print('Phase: train_fp32')
        _ = train_fp32(model_fp32,
                       X_train,
                       y_train,
                       X_val,
                       y_val,
                       save_path=cfp32.save_path,
                       cdr_args=cfp32.train.cdr_args,
                       bsz=cfp32.train.bsz,
                       epochs=cfp32.train.epochs,
                       acc_thres=cfp32.train.acc_thres,
                       )

    set_seed(conf.seed)
    if 'all' in args.run or 'train_fhq' in args.run:
        ckpt_fp32 = args.ckpt_fp32 or get_best_ckpt(Path(conf.fp32.save_path) / 'ckpts')
        print(f'Using checkpoint: {ckpt_fp32}')
        model_fp32.load_weights(ckpt_fp32)

        print('Phase: train_fhq')
        _ = train_fhq(model_fhq,
                      model_fp32,
                      X_train,
                      y_train,
                      X_val,
                      y_val,
                      X_test,
                      y_test,
                      save_path=cfhq.save_path,
                      cdr_args=cfhq.train.cdr_args,
                      bsz=cfhq.train.bsz,
                      epochs=cfhq.train.epochs,
                      acc_thres=cfhq.train.acc_thres
                      )

    bops_computed = False
    ckpt_fhq = args.ckpt_fhq or get_best_ckpt(Path(cfhq.save_path) / 'ckpts')
    if 'all' in args.run or 'test' in args.run:
        print('Phase: test')
        print(f'Using checkpoint: {ckpt_fhq}')
        test(model_fhq, ckpt_fhq, cfhq.save_path, X_train, X_val, X_test, y_test)
        bops_computed = True

    if 'all' in args.run or 'syn' in args.run:
        print('Phase: syn')
        print(f'Using checkpoint: {ckpt_fhq}')
        if not bops_computed:
            print('Computing BOPS...')
            model_fhq.load_weights(ckpt_fhq)
            _ = compute_bops(model_fhq, X_train, bsz=2048, verbose=False)
            bops = compute_bops(model_fhq, X_val, bsz=2048, rst=False)
            print(f'BOPS: {bops}')
        syn_test(model_fhq, ckpt_fhq, cfhq.save_path, X_test, y_test)
