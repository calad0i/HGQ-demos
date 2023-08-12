import numpy as np

import tensorflow as tf
from tensorflow import keras

layers = keras.layers
Input = layers.Input
Reshape = layers.Reshape
Conv1D = layers.Conv1D
Dense = layers.Dense
Add = layers.Add
BatchNormalization = layers.BatchNormalization


class Diag(tf.keras.constraints.Constraint):
    def __init__(self, mask):
        self.mask = tf.constant(mask)

    def __call__(self, W):
        return W * self.mask

    def get_config(self):
        return {'mask': self.mask}


def get_model_fp32(mask12, mask13, mask23):

    relu = keras.layers.ReLU
    act = keras.layers.Activation

    input_m1 = Input(shape=(50, 3), name='M1')
    input_m2 = Input(shape=(50, 2), name='M2')
    input_m3 = Input(shape=(50, 2), name='M3')

    m1_c = act('tanh', name='act1')(Reshape((50,))(Conv1D(1, 3, strides=1, padding='same', activation=None, name='m1_conv', use_bias=False)(input_m1)))
    m2_c = Reshape((50,))(Conv1D(1, 3, strides=1, padding='same', activation=None, name='m2_conv', use_bias=False)(input_m2))
    m3_c = Reshape((50,))(Conv1D(1, 3, strides=1, padding='same', activation=None, name='m3_conv', use_bias=False)(input_m3))

    m1_o = (Dense(50, name='map12', kernel_constraint=Diag(mask12) if mask12 is not None else None, activation=None, use_bias=False)(m1_c))
    m2_i = act('tanh', name='act12')(Add(name='add12')([m1_o, m2_c]))

    m2_o = (Dense(50, name='map23', kernel_constraint=Diag(mask23) if mask23 is not None else None, activation=None, use_bias=False)(m2_i))
    m3_i = (Add(name='add23')([m2_o, m3_c]))

    m1_oo = (Dense(50, name='map13', kernel_constraint=Diag(mask13) if mask13 is not None else None, activation=None, use_bias=False)(m1_c))
    m3_o = (Add(name='add13')([m1_oo, m3_c]))
    m3_o = (Add(name='add23')([m2_o, m3_o]))

    feature_out = act('tanh', name='actf')(m3_o)

    dd1 = Dense(28, activation=None, name='t1')(feature_out)
    dd1 = relu()(BatchNormalization()(dd1))

    dd1 = (Dense(14, activation=None, name='t2')(dd1))
    dd1 = relu()(BatchNormalization()(dd1))

    dd1 = (Dense(8, activation=None, name='t3')(dd1))
    dd1 = relu()(BatchNormalization()(dd1))

    dd1 = Dense(1, bias_initializer=keras.initializers.Constant(229), name='theta_out')(dd1)  # type: ignore

    output = dd1

    model = keras.Model([input_m1, input_m2, input_m3], output, name='TGCNN')
    return model


from HGQ.layers import Signature, HAdd, HConv1D, HDense, HActivation
from HGQ.layers import PReshape
from HGQ.utils import L1
from HGQ import set_default_kernel_quantizer_config, set_default_pre_activation_quantizer_config


def get_model_hgq(mask12, mask13, mask23, conf):

    beta = conf.beta
    l1_cc = conf.l1_cc
    l1_dc = conf.l1_dc
    l1_act = conf.l1_act
    init_bw_k = conf.init_bw_k
    init_bw_a = conf.init_bw_a

    input_m1 = Input(shape=(50, 3), name='_M1')
    input_m2 = Input(shape=(50, 2), name='_M2')
    input_m3 = Input(shape=(50, 2), name='_M3')

    def signature(name):
        kn = np.zeros((1, 50, 1), dtype=np.int8)
        int_bits = np.ones((1, 50, 1), dtype=np.int8)
        bits = np.ones((1, 50, 1), dtype=np.int8)
        return Signature(kn, int_bits, bits, name=name)

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
        rnd_strategy='auto',  # 'auto': 'floor' for layer without bias, 'standard_round' otherwise
        exact_q_value=False,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=L1(l1_act),
        minmax_record=True
    )

    act_q_conf_hg = dict(
        init_bw=init_bw_k,
        skip_dims='batch',
        rnd_strategy='auto',  # 'auto': 'floor' for non-activation layer without bias, 'standard_round' otherwise
        exact_q_value=False,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=L1(l1_act),
        minmax_record=True
    )

    set_default_pre_activation_quantizer_config(act_q_conf)

    aio_c = {
        'beta': beta,
        'kernel_quantizer_config': ker_q_conf_c,
        'activation': None,
        'use_bias': False,
        'padding': 'same',
        'parallel_factor': 50,
    }

    aio_d = {
        'beta': beta,
        'kernel_quantizer_config': ker_q_conf_d
    }

    _input_m1 = signature('M1')(input_m1)
    _input_m2 = signature('M2')(input_m2)
    _input_m3 = signature('M3')(input_m3)

    m1_c = HConv1D(1, 3, strides=1, name='m1_conv', **aio_c, pre_activation_quantizer_config=act_q_conf_hg)(_input_m1)
    m2_c = HConv1D(1, 3, strides=1, name='m2_conv', **aio_c, pre_activation_quantizer_config=act_q_conf_hg)(_input_m2)
    m3_c = HConv1D(1, 3, strides=1, name='m3_conv', **aio_c, pre_activation_quantizer_config=act_q_conf_hg)(_input_m3)

    m1_c = PReshape((50,))(m1_c)
    m2_c = PReshape((50,))(m2_c)
    m3_c = PReshape((50,))(m3_c)

    m1_a = HActivation('tanh', beta=beta, name='act1')(m1_c)
    m1_o = HDense(50, name='map12', kernel_constraint=Diag(mask12) if mask12 is not None else None, activation=None, use_bias=False, **aio_d)(m1_a)

    m2_a = HActivation('tanh', beta=beta, name='act12')(HAdd(name='add12', beta=beta)([m1_o, m2_c]))
    m2_o = HDense(50, name='map23', kernel_constraint=Diag(mask23) if mask12 is not None else None, activation=None, use_bias=False, **aio_d)(m2_a)

    m1_o2 = HDense(50, name='map13', kernel_constraint=Diag(mask13) if mask12 is not None else None, activation=None, use_bias=False, **aio_d)(m1_a)
    m3_o = (HAdd(name='add13', beta=beta)([m1_o2, m3_c]))
    m3_o = (HAdd(name='add23', beta=beta)([m2_o, m3_o]))

    feature_out = HActivation('tanh', beta=beta, name='feature_out')(m3_o)

    dd1 = HDense(28, name='t1', **aio_d, activation='relu', pre_activation_quantizer_config=act_q_conf_hg)(feature_out)
    dd1 = HDense(14, name='t2', **aio_d, activation='relu', pre_activation_quantizer_config=act_q_conf_hg)(dd1)
    dd1 = HDense(8, name='t3', **aio_d, activation='relu', pre_activation_quantizer_config=act_q_conf_hg)(dd1)

    dd1 = HDense(1, name='theta_out')(dd1)

    output = dd1

    model = keras.Model([input_m1, input_m2, input_m3], output, name='TGCNN')
    return model
