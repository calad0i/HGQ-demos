import numpy as np

import tensorflow as tf
import keras
from keras.layers import Reshape, BatchNormalization, Dense, ReLU


def get_model(
    conf,
):

    n = 3 if conf.pt_eta_phi else 16

    inp = keras.layers.Input((conf.n_constituents, n))
    x = Reshape((-1,))(inp)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dense(5)(x)
    out = BatchNormalization()(x)

    model = keras.Model(inputs=inp, outputs=out)

    return model
