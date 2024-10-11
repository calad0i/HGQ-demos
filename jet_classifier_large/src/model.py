import numpy as np

import tensorflow as tf
from tensorflow import keras

from HGQ.utils import MonoL1
from HGQ import set_default_paq_conf, set_default_kq_conf
from HGQ import HConv2D, HDense, HQuantize, PMaxPool2D, PReshape, PFlatten
from HGQ.layers.dense import HDenseBatchNorm
from HGQ.layers.conv import HConv2DBatchNorm
from HGQ.layers.passive_layers import PPermute


def get_model(
    conf,
):

    ker_q_conf = dict(
        init_bw=conf.model.init_bw_a,
        skip_dims=None,
        rnd_strategy='standard_round',
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=MonoL1(conf.model.k_bw_l1_reg),
    )

    act_q_conf = dict(
        init_bw=conf.model.init_bw_k,
        skip_dims=(0,),
        rnd_strategy='auto',  # 'auto': 'floor' for layer without bias except HActivation layers, 'standard_round' otherwise
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=MonoL1(conf.model.a_bw_l1_reg),
        minmax_record=True
    )

    set_default_paq_conf(act_q_conf)
    set_default_kq_conf(ker_q_conf)

    inp = keras.layers.Input((conf.n_constituents, 16))
    x = HQuantize(beta=0)(inp)
    # x = HBatchNormalization()(x)
    # inp_q_mask = MaskDropout(0.1)(inp_q_mask)

    x = HDenseBatchNorm(24, activation='relu', beta=0)(x)
    x = HDenseBatchNorm(16, activation='relu', beta=0)(x)

    x = PPermute((2, 1))(x)
    x = HDenseBatchNorm(conf.n_constituents, activation='relu', beta=0)(x)
    x = PPermute((2, 1))(x)

    x = HDenseBatchNorm(24, activation='relu', beta=0)(x)
    x = HDenseBatchNorm(16, activation='relu', beta=0)(x)

    x = PPermute((2, 1))(x)
    x = HDenseBatchNorm(1, activation='relu', beta=0)(x)
    x = PReshape((16,))(x)

    x = HDenseBatchNorm(24, activation='relu', beta=0)(x)
    x = HDenseBatchNorm(16, activation='relu', beta=0)(x)
    x = HDenseBatchNorm(16, activation='relu', beta=0)(x)
    out = HDenseBatchNorm(5, activation='relu', beta=0)(x)

    model = keras.Model(inputs=inp, outputs=out)

    return model
