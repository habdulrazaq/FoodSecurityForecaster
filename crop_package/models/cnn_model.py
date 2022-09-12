from tensorflow.keras import layers, models, losses


def cnn():

    norm_layer = layers.Normalization()

    model = models.Sequential([
        norm_layer,
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Dropout(0.3),
        layers.LayerNormalization(),
        layers.MaxPooling2D(strides=(2, 2), pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.Dropout(0.3),
        layers.LayerNormalization(),
        layers.MaxPooling2D(strides=(2, 2), pool_size=(2, 2)),
        layers.Flatten(),
        layers.LayerNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss=losses.mse, optimizer='adam')

    return model, norm_layer
