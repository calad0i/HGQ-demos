import sys
sys.path.append('../')


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import mplhep as hep
hep.style.use(hep.style.CMS)

from pathlib import Path

from nn_utils import set_seed, get_best_ckpt
from src.dataloader import get_data
from src.model import get_model
from src.train import train
from src.test import test
from src.syn_test import syn_test

import omegaconf
import argparse

from HGQ.bops import trace_minmax

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--run', '-r', nargs='*', type=str, default=['all'])
    parser.add_argument('--softmax', '-s', action='store_true', help='Add final softmax layer to the model. For comparasion only.')
    args = parser.parse_args()

    path = Path(args.config)

    conf = omegaconf.OmegaConf.load(path)
    uniform = hasattr(conf.model, 'uniform') and conf.model.uniform

    seed = conf.seed

    data_path = Path(conf.data_path)
    save_path = Path(conf.save_path)

    set_seed(seed)

    X_train_val, X_test, y_train_val, y_test = get_data(data_path, mmap_location='/gpu:0', seed=seed)

    model = get_model(conf)

    bops = trace_minmax(model, X_train_val, bsz=664000)
    print(f'Init BOPS: {bops}')

    if 'train' in args.run or 'all' in args.run:
        print('Phase: train')
        train(model, X_train_val, y_train_val, save_path, conf)

    if 'test' in args.run or 'all' in args.run:
        print('Phase: test')
        test(model, save_path, X_train_val, X_test, y_test)

    if 'syn' in args.run or 'all' in args.run:
        print('Phase: syn')
        syn_test(save_path, X_test, y_test, N=None, softmax=args.softmax)
