from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D


def get_FFNN(len_of_last_layer):
    model = Sequential()
    # определяем архитектуру нейронный сети 3072-1024-512-3 (3072=32*32*3)
    model.add(Dense(1024, input_shape=(3072,), activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(len_of_last_layer, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

    model.summary()

    return model


def get_CNN(len_of_last_layer):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(len_of_last_layer, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])

    model.summary()

    return model
