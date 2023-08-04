import numpy as np

import tensorflow as tf
from tensorflow import keras

from FHQ.utils import L1
from FHQ import set_default_kernel_quantizer_config, set_default_pre_activation_quantizer_config
from FHQ import HConv2D, HDense, HQuantize, PMaxPool2D, PReshape, PFlatten


def get_model_fp32(renorm=False):
    def cfg(): return {'kernel_regularizer': keras.regularizers.l1(1e-4), 'kernel_initializer': 'lecun_uniform'}

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(32, 32, 3)),

        keras.layers.Conv2D(16, (3, 3), padding='valid', use_bias=False, **cfg(), name='conv1'),
        keras.layers.BatchNormalization(renorm=renorm),
        keras.layers.MaxPool2D((2, 2), name='maxpool1'),
        keras.layers.ReLU(),

        keras.layers.Conv2D(16, (3, 3), padding='valid', use_bias=False, **cfg(), name='conv2'),
        keras.layers.BatchNormalization(renorm=renorm),
        keras.layers.MaxPool2D((2, 2), name='maxpool2'),
        keras.layers.ReLU(),

        keras.layers.Conv2D(24, (3, 3), padding='valid', use_bias=False, **cfg(), name='conv3'),
        keras.layers.BatchNormalization(renorm=renorm),
        keras.layers.MaxPool2D((2, 2), name='maxpool3'),
        keras.layers.ReLU(),

        keras.layers.Flatten(),
        # keras.layers.Dropout(0.5),

        keras.layers.Dense(42, use_bias=False, **cfg(), name='dense1'),
        keras.layers.BatchNormalization(renorm=renorm),
        keras.layers.ReLU(),

        keras.layers.Dense(64, use_bias=False, **cfg(), name='dense2'),
        keras.layers.BatchNormalization(renorm=renorm),
        keras.layers.ReLU(),

        keras.layers.Dense(10, **cfg(), name='output'),
    ])
    return model


def get_model_fhq(
    init_bw_k=4,
    init_bw_a=4,
    bops_reg_factor=1e-5,
    parallel_factors=(1, 1, 1),
    l1_cc=2e-6,
    l1_dc=2e-6,
    l1_act=2e-6,
):

    ker_q_conf_c = dict(
        init_bw=init_bw_k,
        skip_dims=None,
        rnd_strategy='standard_round',
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=L1(l1_cc),
    )

    ker_q_conf_d = dict(
        init_bw=init_bw_a,
        skip_dims=None,
        rnd_strategy='standard_round',
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=L1(l1_dc),
    )

    act_q_conf = dict(
        init_bw=init_bw_k,
        skip_dims='all',
        rnd_strategy='auto',  # 'auto': 'floor' for layer without bias except HActivation layers, 'standard_round' otherwise
        exact_q_value=False,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=L1(l1_act),
        minmax_record=True
    )

    set_default_pre_activation_quantizer_config(act_q_conf)

    def cfg(): return {
        'kernel_regularizer': keras.regularizers.l1(1e-4),
        'kernel_initializer': 'lecun_uniform',
        'bops_reg_factor': bops_reg_factor,
        'pre_activation_quantizer_config': act_q_conf,
    }

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(32, 32, 3)),
        HQuantize(bops_reg_factor=bops_reg_factor, name='q_inp'),

        HConv2D(16, (3, 3), padding='valid', name='conv1', activation='relu', **cfg(), parallel_factor=parallel_factors[0], kernel_quantizer_config=ker_q_conf_c),
        PMaxPool2D((2, 2), name='maxpool1'),

        HConv2D(16, (3, 3), padding='valid', name='conv2', activation='relu', **cfg(), parallel_factor=parallel_factors[1], kernel_quantizer_config=ker_q_conf_c),
        PMaxPool2D((2, 2), name='maxpool2'),

        HConv2D(24, (3, 3), padding='valid', name='conv3', activation='relu', **cfg(), parallel_factor=parallel_factors[2], kernel_quantizer_config=ker_q_conf_c),
        PMaxPool2D((2, 2), name='maxpool3'),

        PFlatten(),

        HDense(42, name='dense1', activation='relu', **cfg(), kernel_quantizer_config=ker_q_conf_d),

        HDense(64, name='dense2', activation='relu', **cfg(), kernel_quantizer_config=ker_q_conf_d),

        HDense(10, name='output', **cfg()),
    ])

    for l in model.layers:
        if isinstance(l, PMaxPool2D):
            l.last_layer.pre_activation_quantizer.degeneracy /= int(np.prod(l.pool_size))  # pooling layers modifies the exposure of the previous layer
        if isinstance(l, HConv2D):
            l.last_layer.pre_activation_quantizer.degeneracy *= float(l.parallel_factor / l.total_channels)

    return model
