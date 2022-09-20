from tensorflow.keras import layers, models, losses


def cnn_paper():

    norm_layer = layers.Normalization()

    model = models.Sequential([
        norm_layer,
        layers.Conv2D(32, kernel_size=(1,2), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(strides=(1, 2), pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(1,2), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(strides=(1, 2), pool_size=(2, 2)),
        layers.Flatten(),
        layers.BatchNormalization(),

        layers.Dense(1, activation='linear')
    ])

    model.compile(loss=losses.mse, optimizer='adam')

    return model, norm_layer
