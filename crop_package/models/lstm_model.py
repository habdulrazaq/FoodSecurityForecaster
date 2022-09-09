from tensorflow.keras import layers, models, losses
from tensorflow.keras import optimizers


def lstm():

    norm_layer = layers.Normalization(input_shape=(46, 30, 8))
    #(14, 46, 30, 8)
    model = models.Sequential([
        norm_layer,
        layers.Reshape((46, 30 * 8)),
        layers.LSTM(units=128, activation='tanh', recurrent_dropout=0.1),
        layers.LayerNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='relu'),
    ])

    optimizer=optimizers.Adam(learning_rate=5e-4)
    model.compile(loss=losses.mse, optimizer=optimizer)

    return model, norm_layer
