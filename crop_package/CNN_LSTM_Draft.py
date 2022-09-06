from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def CNN_LSTM():

    #Model instantiation
    model = Sequential()

   ########################
   # 1. CNN ARCHITECTURE  #
   ########################

   #Conv2D Layer 1
    model.add(layers.Conv2D(32, kernel_size=(1, 2), activation="relu", input_shape=X_train_small[0].shape))
   #Batch Normalization Layer 1
    model.add(BatchNormalization())
   #Max-Pooling 2D Layer 1
    model.add(layers.MaxPooling2D(kernel_size=(1,2), pool_size=(2, 2))
   #Conv2D Layer 2
    model.add(layers.Conv2D(64, kernel_size=(1, 2), activation="relu", input_shape=X_train_small[0].shape))
   #Batch Normalization Layer 2
    model.add(BatchNormalization())
   #Max-Pooling 2D Layer 2
    model.add(layers.MaxPooling2D(kernel_size=(1,2), pool_size=(2, 2))

   #Flattening Layer
    model.add(Flatten())
   #Final Batch Normalization Layer in CNN
    model.add(BatchNormalization())

   #########################
   # 2. LSTM ARCHITECTURE  #
   #########################

   #LSTM Layer with 256 Neurons
    model_LSTM.add(LSTM(units=256, activation='tanh'))
   #Dense Layer with 64 Neurons
    model_LSTM.add(Dense(64, activation='relu'))
   #Flattening Layer
    model.add(Flatten())
   #Dropout Layer
    model.add(Dropout(0.5))
   #Predictive output Layer
    model_LSTM.add(Dense(1, activation='relu'))

   #########################
   # 3. Compiler           #
   #########################
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
