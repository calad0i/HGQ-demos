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
from src.train import train
from src.test import test
from src.syn_test import syn_test

import omegaconf
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--run', '-r', nargs='*', type=str, default=['all'])
    parser.add_argument('--ckpt', '-k', type=str, default=None)
    args = parser.parse_args()

    path = Path(args.config)

    conf = omegaconf.OmegaConf.load(path)

    seed = conf.seed

    data_path = Path(conf.data_path)
    save_path = Path(conf.save_path)

    set_seed(seed)

    X_train_val, X_test, y_train_val, y_test = get_data(data_path, mmap_location='/gpu:0', seed=seed)

    model = get_model(conf.model.bops_reg_factor, conf.model.a_bw_l1_reg, conf.model.w_bw_l1_reg, conf.model.a_init_bw, conf.model.w_init_bw)

    from HGQ.bops import compute_bops
    bops = compute_bops(model, X_train_val, bsz=664000)
    print(f'Init BOPS: {bops}')

    if 'train' in args.run or 'all' in args.run:
        print('Phase: train')
        train(model,
              X_train_val,
              y_train_val,
              save_path,
              lr=conf.train.lr,
              epochs=conf.train.epochs,
              bsz=conf.train.bsz,
              val_split=conf.train.val_split,
              acc_thres=conf.train.acc_thres,
              )

    bops_computed = False
    if 'test' in args.run or 'all' in args.run:
        print('Phase: test')
        ckpt_path = args.ckpt or get_best_ckpt(save_path / 'ckpts')
        print(f'Using checkpoint: {ckpt_path}')
        test(model, Path(ckpt_path), save_path, X_train_val, X_test, y_test)
        bops_computed = True

    if 'syn' in args.run or 'all' in args.run:
        print('Phase: syn')
        ckpt_path = args.ckpt or get_best_ckpt(save_path / 'ckpts')
        if not bops_computed:
            print('Computing BOPS...')
            model.load_weights(ckpt_path)
            bops = compute_bops(model, X_train_val, bsz=664000)
            print(f'BOPS: {bops}')
        print(f'Using checkpoint: {ckpt_path}')
        syn_test(model, Path(ckpt_path), save_path, X_test, y_test, N=None)
