import tensorflow as tf
from tensorflow import keras

from HGQ import HDense, HQuantize
from HGQ.layers.dense import HDenseBatchNorm
from HGQ import set_default_paq_conf, set_default_kq_conf
from HGQ.utils import MonoL1


def get_model(conf):

    ker_q_conf: dict = dict(
        init_bw=conf.model.w_init_bw,
        skip_dims=None if not conf.model.uniform_w else 'all',
        rnd_strategy='standard_round',
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=MonoL1(conf.model.w_bw_l1_reg),
    )

    act_q_conf: dict = dict(
        init_bw=conf.model.a_init_bw,
        skip_dims=(0,) if not conf.model.uniform_a else 'all',
        rnd_strategy='auto',  # 'auto': 'floor' for layer without bias, 'standard_round' otherwise
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=MonoL1(conf.model.a_bw_l1_reg),
        minmax_record=True
    )

    set_default_kq_conf(ker_q_conf)
    set_default_paq_conf(act_q_conf)

    hidden_layers = [
        HDenseBatchNorm(round(64 * conf.model.scale), activation='relu', name='dense_1', beta=0),
        HDenseBatchNorm(round(32 * conf.model.scale), activation='relu', name='dense_2', beta=0),
        HDenseBatchNorm(round(32 * conf.model.scale), activation='relu', name='dense_3', beta=0),
        HDenseBatchNorm(round(32 * conf.model.scale), activation='relu', name='dense_extra', beta=0),
    ]

    if conf.model.n_hidden_layers > 0:
        hidden_layers = hidden_layers[-conf.model.n_hidden_layers:]
    else:
        hidden_layers = []

    q_layer = HQuantize(input_shape=(16,), name='inp_q', beta=0)
    hat = HDenseBatchNorm(5, name='dense_4', beta=0)

    layers = [q_layer] + hidden_layers + [hat]

    model = keras.Sequential(layers)

    return model
