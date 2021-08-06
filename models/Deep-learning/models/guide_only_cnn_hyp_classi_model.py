import tensorflow as tf
from kerastuner import HyperParameters
from tensorflow import keras

from models.layers import recurrent_dense, strided_down, encoder_down_block


def guide_only_cnn_hyp_classi_model(num_strided_down=4,kernel=5,cnn_units=128, dense_units=128, recurrent_layers=8, noise=True):
    seq = keras.Input(shape=(30, 4))
    #other = keras.Input(shape=9)

    x = seq
    for _ in range(num_strided_down):
        x = strided_down(x, cnn_units, 1, kernel)
        if noise:
            x = keras.layers.GaussianNoise(.01)(x)

    x = keras.layers.Flatten()(x)

    #x = keras.layers.Concatenate()([x, other])

    x = keras.layers.Dense(dense_units, activation=tf.nn.leaky_relu)(x)
    for _ in range(recurrent_layers):
        x = recurrent_dense(x, dense_units)

    outputs = keras.layers.Dense(1)(x)
    # TODO: make a second output that is confidence, and have some allowance of reduced penalty
    #  for low confidence wrong guesses, but overall penalty for low confidence
    return keras.Model(inputs=[seq], outputs=outputs)


def guide_only_cnn_hyp_classi_model_hp(hp: HyperParameters):
    kernel= hp.Choice('kernel',[3,4,5])
    cnn_units = hp.Choice('cnn_units', [8,16,32,64])
    dense_units = hp.Choice('dense_units', [8,16,32,64])
    num_strided_down = hp.Int('num_strided_down', 3,5)
    recurrent_layers = hp.Choice('num_recurrent_layers', [0,1,2,3])
    noise = True # hp.Boolean('use_noise')

    model = guide_only_cnn_hyp_classi_model(num_strided_down=num_strided_down, kernel=kernel, cnn_units=cnn_units, dense_units=dense_units,
                             recurrent_layers=recurrent_layers, noise=noise)

    #metrics = [keras.metrics.MeanAbsoluteError(), keras.metrics.MeanSquaredError()]

    metrics = ['accuracy']

    model.compile(keras.optimizers.Adam(), tf.losses.binary_crossentropy, metrics=metrics)
    #model.compile(keras.optimizers.Adam(), tf.keras.losses.MeanSquaredError(), metrics=metrics)

    return model
