from keras.datasets import mnist
import numpy as np
from keras import utils

def get_mnist():
    """
    prepare mnist data
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    X_test = X_test.astype(np.float32) / 255.

    y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

    return (X_train, y_train), (X_test, y_test)