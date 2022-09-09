import numpy as np
from tensorflow.keras  import models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

from models.cnn_model import cnn
from models.lstm_model import lstm
from models.baseline_model import BaselineModel
import USA_get_X_y

import matplotlib.pyplot as plt

def fit_model(build_model=cnn):

    model, norm_layer = build_model()

    X, y = USA_get_X_y.get_X_y()

    X_train, X_test, y_train, y_test = USA_get_X_y.split_years(X, y, test_size=2)

    X_train, X_valid, y_train, y_valid = USA_get_X_y.split_years(X_train, y_train, test_size=2)

    es = EarlyStopping(monitor = "val_loss",
                       patience = 500,
                       mode = "min",
                       restore_best_weights = True)

    baseline = BaselineModel()
    baseline.fit(X_train, y_train)
    y_baseline = baseline.predict(X_test)
    baseline_score = mean_squared_error(y_test, y_baseline)

    print(X.shape, X_train.shape)
    X_train = X_train.reshape((-1,) + X_train.shape[2:])
    X_valid = X_valid.reshape((-1,) + X_valid.shape[2:])
    X_test = X_test.reshape((-1,) + X_test.shape[2:])

    y_train = y_train.flatten()
    y_valid = y_valid.flatten()
    y_test = y_test.flatten()

    print(X_train.shape)
    print(y_train.shape)
    norm_layer.adapt(X_train)
    history = model.fit(X_train, y_train,
                        validation_data = (X_valid, y_valid),
                        shuffle = True,
                        batch_size = 32,
                        epochs = 500,
                        callbacks=[es],
                        verbose = 1)

    print("cnn test rmse:", model.evaluate(X_test, y_test) ** 0.5)
    print("baseline test rmse:", baseline_score ** 0.5)

    return model, history

model, history = fit_model(lstm)


t = np.arange(len(history.history['loss']))
plt.semilogy(t, history.history['loss'], label='train loss')
plt.semilogy(t, history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
