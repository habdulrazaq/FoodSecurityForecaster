import numpy as np
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

from models.cnn_model import cnn
from models.baseline_model import BaselineModel
import get_X_y

import matplotlib.pyplot as plt

def fit_model(build_model=cnn):

    model = build_model()

    X, y = get_X_y.get_X_y()

    X_train, X_test, y_train, y_test = get_X_y.split_years(X, y, test_size=2)

    X_train, X_valid, y_train, y_valid = get_X_y.split_years(X_train, y_train, test_size=2)

    es = EarlyStopping(monitor = "val_loss",
                       patience = 30,
                       mode = "min",
                       restore_best_weights = True)

    y_baseline = BaselineModel().fit(X_train, y_train).predict(X_test)
    baseline_score = mean_squared_error(y_test, y_baseline)

    X_train = X_train.reshape((-1,) + X_train.shape[2:])
    X_valid = X_valid.reshape((-1,) + X_valid.shape[2:])
    X_test = X_test.reshape((-1,) + X_test.shape[2:])

    y_train = y_train.flatten()
    y_valid = y_valid.flatten()
    y_test = y_test.flatten()

    history = model.fit(X_train, y_train,
                        validation_data = (X_valid, y_valid),
                        shuffle = True,
                        batch_size = 32,
                        epochs = 500,
                        verbose = 1)

    print("cnn test rmse:", model.evaluate(X_test, y_test) ** 0.5)
    print("baseline test rmse:", baseline_score ** 0.5)

    return model, history

model, history = fit_model()


t = np.arange(len(history.history['loss']))
plt.semilogy(t, history.history['loss'], label='train loss')
plt.semilogy(t, history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
