from cnn_model import cnn
from tensorflow.keras import models

def fit_model(model=cnn):

    es = EarlyStopping(monitor = "val_loss",
                      patience = 3,
                      mode = "min",
                      restore_best_weights = True)

    history = model.fit(X_train, y_train,
                        validation_split = 0.2,
                        shuffle = False,
                        batch_size = 32,
                        epochs = 50,
                        callbacks = [es],
                        verbose = 1)

    return history
