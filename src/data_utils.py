import numpy as np
from keras.datasets import mnist
from keras import utils
from sklearn.metrics import accuracy_score


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


def get_concrete(path):
    data = np.loadtxt(path)
    X_train = data[:,:8]
    y_train = data[:,8]
    return X_train, y_train


def split_to_create_db(X_train, y_train, fold_size=0.2):
    db_samples = int(fold_size * X_train.shape[0])
    X_train_db = X_train[:db_samples]
    y_train_db = y_train[:db_samples]
    X_train = X_train[db_samples:]
    y_train = y_train[db_samples:]
    return X_train, X_train_db, y_train, y_train_db
    

def probs_accuracy(y_probs, y_pred_categorical):
    y_probs_categorical = utils.to_categorical(y_probs.argmax(axis=1))
    return accuracy_score(y_probs_categorical, y_pred_categorical)