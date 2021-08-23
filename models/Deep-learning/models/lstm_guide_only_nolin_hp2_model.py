import tensorflow as tf
from tensorflow import keras

from models.layers import recurrent_dense


def lstm_guide_only_nolin_hp2_model(args,lstm_units=64, dense_units=16, recurrent_layers=1, dropout=0.0):
    seq = keras.Input(shape=(None, 4)) # 4 

    x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units, dropout=dropout))(seq)
    x = keras.layers.Dense(dense_units)(x)

    for _ in range(recurrent_layers):
        x = recurrent_dense(x, dense_units)

    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs=[seq], outputs=outputs)

