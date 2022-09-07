from tensorflow.keras import layers, models, losses


def lstm():

    model = models.Sequential([
        layers.LSTM(units=256, activation='tanh'))
        layers.add(Dense(64, activation='relu'))
        layers.add(Flatten())
        layers.add(Dropout(0.5))
        layers.add(Dense(1, activation='relu'))
    ])

    model.compile(loss=losses.mse, optimizer='adam')

    return model
