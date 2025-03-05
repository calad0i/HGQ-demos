import numpy as np

import tensorflow as tf
import keras

from keras.layers import Dense, BatchNormalization, ReLU, Permute, Reshape, Add
from HGQ.layers import HAdd

def get_model(N,n):

    inp = keras.layers.Input((N, n))
    x = inp
    x2 = inp
    # x = HBatchNormalization()(x)
    # inp_q_mask = MaskDropout(0.1)(inp_q_mask)

    x = ReLU()((Dense(16)(x)))
    x = ReLU()((Dense(n)(x)))

    x = Permute((2, 1))(x)
    x = ReLU()((Dense(N)(x)))
    x = Permute((2, 1))(x)

    x3 = Add()([x, x2])

    x = ReLU()((Dense(16)(x3)))
    x = ReLU()((Dense(16)(x)))
    
    # x = HAdd(beta=0)([x, x3])

    x = Permute((2, 1))(x)
    x = ReLU()((Dense(1)(x)))
    x = Reshape((16,))(x)

    x = ReLU()((Dense(16)(x)))
    x = ReLU()((Dense(16)(x)))
    x = ReLU()((Dense(16)(x)))
    out = (Dense(16)(x))

    model = keras.Model(inputs=inp, outputs=out)

    return model
