import numpy as np
from tensorflow.keras  import models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

from models.cnn_model import cnn
from models.lstm_model import lstm
from models.baseline_model import BaselineModel
import USA_get_X_y

import matplotlib.pyplot as plt

import sys

def fit_model(build_model=cnn, verbose=True):

    model, norm_layer = build_model()

    X, y, _ = USA_get_X_y.get_X_y()

    if verbose:
        plt.scatter(np.arange(y.flatten().shape[0]), y.flatten())
        plt.show()

    X_train, X_test, y_train, y_test = USA_get_X_y.split_years(X, y, test_size=2)

    X_train, X_valid, y_train, y_valid = USA_get_X_y.split_years(X_train, y_train, test_size=2)

    es = EarlyStopping(monitor = "val_loss",
                       patience = 50,
                       mode = "min",
                       restore_best_weights = True)

    X_test_full_year = X_test[:,:,:1]
    y_test_full_year = y_test[:,:,:1]


    baseline = BaselineModel()
    baseline.fit(X_train, y_train)
    y_baseline = baseline.predict(X_test)
    baseline_r2 = r2_score(y_test.flatten(), y_baseline.flatten())
    baseline_mse = mean_squared_error(y_test.flatten(), y_baseline.flatten())

    if verbose:
        print("baseline R^2:", baseline_r2)
        print("baseline mse:", baseline_mse)

        print(X.shape, X_train.shape)
    X_train = X_train.reshape((-1,) + X_train.shape[-3:])
    X_valid = X_valid.reshape((-1,) + X_valid.shape[-3:])
    X_test = X_test.reshape((-1,) + X_test.shape[-3:])
    X_test_full_year = X_test_full_year.reshape((-1,) + X_test_full_year.shape[-3:])
    X = X.reshape((-1,) + X.shape[-3:])

    y_train = y_train.flatten()
    y_valid = y_valid.flatten()
    y_test = y_test.flatten()
    y_test_full_year = y_test_full_year.flatten()
    y = y.flatten()

    if verbose:
        print(X_train.shape, y_train.shape)

    norm_layer.adapt(X_train)
    history = model.fit(X_train, y_train,
                        validation_data = (X_valid, y_valid),
                        shuffle = True,
                        batch_size = 32,
                        epochs = 5,
                        callbacks=[es],
                        verbose = verbose)
    if verbose:
        plt.scatter(np.arange(len(y)), y, color='green')
        plt.scatter(np.arange(len(y)), model.predict(X), color='red')
        plt.show()
    y_pred = model.predict(X_test).flatten()
    y_pred_full_year = model.predict(X_test_full_year).flatten()


    if verbose:
        print("model R^2:", r2_score(y_pred, y_test.flatten()))
        print("model mse:", mean_squared_error(y_pred, y_test.flatten()))
        print("full year model R^2:", r2_score(y_pred_full_year, y_test_full_year.flatten()))
        print("full year model mse:", mean_squared_error(y_pred_full_year, y_test_full_year.flatten()))
        print("baseline R^2:", baseline_r2)
        print("baseline mse:", baseline_mse)

    return model, history

if __name__ == '__main__':
    model, history = fit_model(cnn)
    if len(sys.argv) > 1:
        model.save(sys.argv[1])
