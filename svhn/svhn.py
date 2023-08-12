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
from src.model import get_model_fp32, get_model_hgq
from src.train import train_fp32, train_hgq
from src.test import test
from src.syn_test import syn_test

from HGQ.bops import compute_bops

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

    data_path = conf.datapath

    print('Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(data_path, val_split=conf.val_split, seed=conf.seed, mmap_location='/gpu:0')

    print('Creating models...')
    model_fp32 = get_model_fp32(renorm=False)

    model_hgq = get_model_hgq(
        init_bw_k=cHGQ.model.a_init_bw,
        init_bw_a=cHGQ.model.k_init_bw,
        beta=cHGQ.model.beta,
        parallel_factors=cHGQ.model.parallel_factors,
        l1_cc=cHGQ.model.k_bw_l1_reg_conv,
        l1_dc=cHGQ.model.k_bw_l1_reg_dense,
        l1_act=cHGQ.model.a_bw_l1_reg,
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
    if 'all' in args.run or 'train_hgq' in args.run:
        ckpt_fp32 = args.ckpt_fp32 or get_best_ckpt(Path(conf.fp32.save_path) / 'ckpts')
        print(f'Using checkpoint: {ckpt_fp32}')
        model_fp32.load_weights(ckpt_fp32)

        print('Phase: train_hgq')
        _ = train_hgq(model_hgq,
                      model_fp32,
                      X_train,
                      y_train,
                      X_val,
                      y_val,
                      X_test,
                      y_test,
                      save_path=cHGQ.save_path,
                      cdr_args=cHGQ.train.cdr_args,
                      bsz=cHGQ.train.bsz,
                      epochs=cHGQ.train.epochs,
                      acc_thres=cHGQ.train.acc_thres
                      )

    bops_computed = False
    ckpt_hgq = args.ckpt_hgq or get_best_ckpt(Path(cHGQ.save_path) / 'ckpts')
    if 'all' in args.run or 'test' in args.run:
        print('Phase: test')
        print(f'Using checkpoint: {ckpt_hgq}')
        test(model_hgq, ckpt_hgq, cHGQ.save_path, X_train, X_val, X_test, y_test)
        bops_computed = True

    if 'all' in args.run or 'syn' in args.run:
        print('Phase: syn')
        print(f'Using checkpoint: {ckpt_hgq}')
        if not bops_computed:
            print('Computing BOPS...')
            model_hgq.load_weights(ckpt_hgq)
            _ = compute_bops(model_hgq, X_train, bsz=2048, verbose=False)
            bops = compute_bops(model_hgq, X_val, bsz=2048, rst=False)
            print(f'BOPS: {bops}')
        syn_test(model_hgq, ckpt_hgq, cHGQ.save_path, X_test, y_test)
