from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
from keras.layers import Dense, LeakyReLU, Dropout, Flatten, Activation, TimeDistributed
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D


def second_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(780, input_shape=(None, input_dim)))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))

    model.add(Dense(780))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))

    model.add(Dense(780))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))

    model.add(Dense(output_dim, activation="softmax"))
    return model


def third_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.15))

    model.add(LSTM(128, return_sequences=True))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.15))

    model.add(TimeDistributed(Dense(output_dim, activation="softmax")))
    return model


def first_model(input_dim, output_dim, kernel_size=32):
    model = Sequential()
    model.add(Conv1D(128, input_shape=(None, input_dim), kernel_size=kernel_size, padding="same", strides=1))
    model.add(Dropout(0.2))
    model.add(LeakyReLU(0.2))
    model.add(Conv1D(64, kernel_size=kernel_size, padding="same", strides=1))
    model.add(Dropout(0.2))
    model.add(LeakyReLU(0.2))

    model.add(Conv1D(32, kernel_size=kernel_size, padding="same", strides=1))
    model.add(Dropout(0.2))
    model.add(LeakyReLU(0.2))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(16))
    model.add(Dense(output_dim, activation="softmax"))
    
    return model

def lstm(input_shape, output_dim, batch_size):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(input_shape[0], input_shape[1])))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.3))

    # model.add(LSTM(128, return_sequences=True))
    # model.add(LeakyReLU(0.05))
    # model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=False))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.3))
    # model.add(Flatten())
    model.add(Dense(output_dim, activation="softmax"))
    return model