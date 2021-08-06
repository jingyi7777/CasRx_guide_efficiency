import tensorflow as tf
from tensorflow import keras
from kerastuner import HyperParameters

from models.layers import recurrent_dense


def guide_only_lstm_hyp_classi_model(lstm_units=16, dense_units=16, recurrent_layers=2, dropout=0.0):
    seq = keras.Input(shape=(None, 4)) # 4 + (3 if linseq)
    #other = keras.Input(shape=5)

    x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units, dropout=dropout))(seq)

    #x = keras.layers.Concatenate()([x, other])
    x = keras.layers.Dense(dense_units)(x)

    for _ in range(recurrent_layers):
        x = recurrent_dense(x, dense_units)

    outputs = keras.layers.Dense(1)(x)
    # TODO: make a second output that is confidence, and have some allowance of reduced penalty
    #  for low confidence wrong guesses, but overall penalty for low confidence
    return keras.Model(inputs=[seq], outputs=outputs)


def guide_only_lstm_hyp_classi_model_hp(hp: HyperParameters):
    lstm_units = hp.Choice('lstm_units', [16, 32,64,128])
    dense_units = hp.Choice('dense_units', [8, 16, 32])
    recurrent_layers = hp.Int('num_recurrent_layers', 0, 3)
    dropout = hp.Choice('dropout', [0.0, 0.1, 0.25])

    model = guide_only_lstm_hyp_classi_model(lstm_units=lstm_units, dense_units=dense_units, recurrent_layers=recurrent_layers,
                              dropout=dropout)

    #metrics = [keras.metrics.MeanAbsoluteError(), keras.metrics.MeanSquaredError()]
    metrics =['accuracy']
    model.compile(keras.optimizers.Adam(), tf.losses.binary_crossentropy, metrics=metrics)
    #model.compile(keras.optimizers.Adam(), tf.keras.losses.MeanSquaredError(), metrics=metrics)

    return model
