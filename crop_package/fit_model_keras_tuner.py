from tensorflow.keras import layers, models, losses, callbacks
import keras_tuner as kt
import get_X_y_USA_INDIA

def hyper_cnn(hp):

    norm_layer = layers.Normalization()

    conv2d_1 = hp.Int('conv2d_1', min_value=2, max_value=120, step=16)
    dropout_1 = hp.Float('dropout_1', min_value=.2, max_value=.5, step=.05)
    conv2d_2 = hp.Int('conv2d_2', min_value=2, max_value=120, step=16)
    dropout_2 = hp.Float('dropout_2', min_value=.2, max_value=.5, step=.05)
    dense_1 = hp.Int('dense_1', min_value=1, max_value = 40, step=1)
    dropout_3 = hp.Float('dropout_3', min_value=.2, max_value=.5, step=.05)

    model = models.Sequential([
        norm_layer,
        layers.Conv2D(conv2d_1, kernel_size=(3, 3), activation="relu"),
        layers.Dropout(dropout_1),
        layers.LayerNormalization(),
        layers.MaxPooling2D(strides=(2, 2), pool_size=(2, 2)),
        layers.Conv2D(conv2d_2, kernel_size=(3, 3), activation="relu"),
        layers.Dropout(dropout_2),
        layers.LayerNormalization(),
        layers.MaxPooling2D(strides=(2, 2), pool_size=(2, 2)),
        layers.Flatten(),
        layers.LayerNormalization(),
        layers.Dense(dense_1, activation='relu'),
        layers.Dropout(dropout_3),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss=losses.mse, optimizer='adam')

    return model

if __name__== "__main__":
    X, y, _ = get_X_y_USA_INDIA.get_X_y()
    X_train, X_test, y_train, y_test = get_X_y_USA_INDIA.split_years(X, y, test_size=2)
    X_train, X_valid, y_train, y_valid = get_X_y_USA_INDIA.split_years(X_train, y_train, test_size=2)
    X_train = X_train.reshape((-1,) + X_train.shape[-3:])
    X_valid = X_valid.reshape((-1,) + X_valid.shape[-3:])
    X_test = X_test.reshape((-1,) + X_test.shape[-3:])


    X = X.reshape((-1,) + X.shape[2:])
    y_train = y_train.flatten()
    y_valid = y_valid.flatten()
    y_test = y_test.flatten()
    y = y.flatten()


    tuner = kt.Hyperband(hyper_cnn,
                     objective='val_loss',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

    es = callbacks.EarlyStopping(monitor='val_loss', patience=100)
    tuner.search(X_train, y_train, epochs=500, validation_data = (X_valid, y_valid), callbacks=[es])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the 1st densely-connected
    layer is {best_hps}.""")
