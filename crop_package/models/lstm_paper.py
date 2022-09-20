from tensorflow.keras import layers, models, losses
from tensorflow.keras.layers import TimeDistributed

def lstm_paper():

    # MAking a model with Time Distributed layers to capture feeding the inputs in one by one
    model = models.Sequential()  #10, 20, 46, 30, 7)
    model.add(TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(10, 20, 46, 30, 7)))
    model.add(TimeDistributed(layers.Conv2D(128, (2, 2), activation='relu', padding='same')))
    model.add(TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(units=32, activation='tanh'))
    model.add(layers.Dense(14))
    model.add(layers.Dense(1, activation = "linear"))
    model.compile(optimizer="Adam", loss = "mse", metrics = "mae")

    model.compile(loss=losses.mse, optimizer='adam')

    return model
