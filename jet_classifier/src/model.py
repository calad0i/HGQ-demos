import tensorflow as tf
from tensorflow import keras

from HGQ import HDense, HQuantize
from HGQ.layers.dense import HDenseBatchNorm
from HGQ import set_default_paq_conf, set_default_kq_conf
from HGQ.utils import MonoL1


def get_model(conf):

    ker_q_conf: dict = dict(
        init_bw=conf.model.w_init_bw,
        skip_dims=None if not conf.model.uniform else 'all',
        rnd_strategy='standard_round',
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=MonoL1(conf.model.w_bw_l1_reg),
    )

    act_q_conf: dict = dict(
        init_bw=conf.model.a_init_bw,
        skip_dims=(0,) if not conf.model.uniform else 'all',
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

    model = keras.Sequential([
        HQuantize(input_shape=(16,), name='inp_q', beta=0),
        HDense(int(64), activation='relu', name='dense_1', beta=0),
        HDense(int(32), activation='relu', name='dense_2', beta=0),
        HDense(int(32), activation='relu', name='dense_3', beta=0),
        HDense(5, name='dense_4', beta=0),
    ])

    return model
