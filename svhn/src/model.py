import numpy as np

import tensorflow as tf
from tensorflow import keras

from HGQ.utils import MonoL1
from HGQ import set_default_paq_conf, set_default_kq_conf
from HGQ import HConv2D, HDense, HQuantize, PMaxPool2D, PReshape, PFlatten
from HGQ.layers.dense import HDenseBatchNorm
from HGQ.layers.conv import HConv2DBatchNorm


# def get_model_fp32(renorm=False):
#     def cfg(): return {'kernel_regularizer': keras.regularizers.l1(1e-4), 'kernel_initializer': 'lecun_uniform'}

#     model = keras.models.Sequential([
#         keras.layers.InputLayer(input_shape=(32, 32, 3)),

#         keras.layers.Conv2D(16, (3, 3), padding='valid', use_bias=False, **cfg(), name='conv1'),
#         keras.layers.BatchNormalization(renorm=renorm),
#         keras.layers.MaxPool2D((2, 2), name='maxpool1'),
#         keras.layers.ReLU(),

#         keras.layers.Conv2D(16, (3, 3), padding='valid', use_bias=False, **cfg(), name='conv2'),
#         keras.layers.BatchNormalization(renorm=renorm),
#         keras.layers.MaxPool2D((2, 2), name='maxpool2'),
#         keras.layers.ReLU(),

#         keras.layers.Conv2D(24, (3, 3), padding='valid', use_bias=False, **cfg(), name='conv3'),
#         keras.layers.BatchNormalization(renorm=renorm),
#         keras.layers.MaxPool2D((2, 2), name='maxpool3'),
#         keras.layers.ReLU(),

#         keras.layers.Flatten(),
#         # keras.layers.Dropout(0.5),

#         keras.layers.Dense(42, use_bias=False, **cfg(), name='dense1'),
#         keras.layers.BatchNormalization(renorm=renorm),
#         keras.layers.ReLU(),

#         keras.layers.Dense(64, use_bias=False, **cfg(), name='dense2'),
#         keras.layers.BatchNormalization(renorm=renorm),
#         keras.layers.ReLU(),

#         keras.layers.Dense(10, **cfg(), name='output'),
#     ])
#     return model


def get_model(
    conf,
):

    ker_q_conf_c = dict(
        init_bw=conf.model.init_bw_k,
        skip_dims=None,
        rnd_strategy='standard_round',
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=MonoL1(conf.model.k_bw_l1_reg_conv),
    )

    ker_q_conf_d = dict(
        init_bw=conf.model.init_bw_a,
        skip_dims=None,
        rnd_strategy='standard_round',
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=MonoL1(conf.model.k_bw_l1_reg_dense),
    )

    act_q_conf = dict(
        init_bw=conf.model.init_bw_k,
        skip_dims='all',
        rnd_strategy='auto',  # 'auto': 'floor' for layer without bias except HActivation layers, 'standard_round' otherwise
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=MonoL1(conf.model.a_bw_l1_reg),
        minmax_record=True
    )

    set_default_paq_conf(act_q_conf)

    def cfg(): return {
        'kernel_regularizer': keras.regularizers.l1(1e-4),
        'kernel_initializer': 'lecun_uniform',
        'paq_conf': act_q_conf,
    }

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(32, 32, 3)),
        HQuantize(beta=0, name='q_inp'),

        HConv2DBatchNorm(16, (3, 3), padding='valid', name='conv1', activation='relu', **cfg(), parallel_factor=conf.model.parallel_factors[0], kq_conf=ker_q_conf_c),
        PMaxPool2D((2, 2), name='maxpool1'),

        HConv2DBatchNorm(16, (3, 3), padding='valid', name='conv2', activation='relu', **cfg(), parallel_factor=conf.model.parallel_factors[1], kq_conf=ker_q_conf_c),
        PMaxPool2D((2, 2), name='maxpool2'),

        HConv2DBatchNorm(24, (3, 3), padding='valid', name='conv3', activation='relu', **cfg(), parallel_factor=conf.model.parallel_factors[2], kq_conf=ker_q_conf_c),
        PMaxPool2D((2, 2), name='maxpool3'),

        PFlatten(),

        HDenseBatchNorm(42, name='dense1', activation='relu', **cfg(), kq_conf=ker_q_conf_d),

        HDenseBatchNorm(64, name='dense2', activation='relu', **cfg(), kq_conf=ker_q_conf_d),

        HDenseBatchNorm(10, name='output', **cfg()),
    ])

    for l in model.layers:
        if isinstance(l, PMaxPool2D):
            l.last_layer.paq.degeneracy /= int(np.prod(l.pool_size))  # pooling layers modifies the exposure of the previous layer
        if isinstance(l, HConv2D):
            l.paq.degeneracy *= float(l.parallel_factor / l.total_channels)

    return model
