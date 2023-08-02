import tensorflow as tf
from tensorflow import keras

from FHQ import HDense, HQuantize
from FHQ import set_default_kernel_quantizer_config, set_default_pre_activation_quantizer_config
from FHQ.utils import L1


def get_model(bops_reg_factor, a_bw_l1_reg=0., w_bw_l1_reg=0., a_init_bw=2, w_init_bw=2):

    ker_q_conf = dict(
        init_bw=w_init_bw,
        skip_dims=None,
        rnd_strategy='standard_round',
        exact_q_value=True,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=L1(w_bw_l1_reg),
    )


    act_q_conf = dict(
        init_bw=a_init_bw,
        skip_dims=(0,),  # same as 'batch'
        rnd_strategy='auto',  # 'auto': 'floor' for layer without bias, 'standard_round' otherwise
        exact_q_value=False,
        dtype=None,
        bw_clip=(-23, 23),
        trainable=True,
        regularizer=L1(2e-6),
        minmax_record=a_bw_l1_reg
    )

    set_default_kernel_quantizer_config(ker_q_conf)
    set_default_pre_activation_quantizer_config(act_q_conf)


    model = keras.Sequential([
        HQuantize(input_shape=(16,), name='inp_q', bops_reg_factor=bops_reg_factor),
        HDense(64, activation='relu', name='dense_1', bops_reg_factor=bops_reg_factor),
        HDense(32, activation='relu', name='dense_2', bops_reg_factor=bops_reg_factor),
        HDense(32, activation='relu', name='dense_3', bops_reg_factor=bops_reg_factor),
        HDense(5, name='dense_4', bops_reg_factor=bops_reg_factor),
    ])

    return model