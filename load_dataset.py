import h5py
import numpy as np


def load_data():

    train_dataset = h5py.File(
        'dataset/train_catvnoncat.h5', "r")
    # training set
    x_train = np.array(train_dataset["train_set_x"][:])
    y_train = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(
        'dataset/test_catvnoncat.h5', "r")

    # testing set
    x_test = np.array(test_dataset["test_set_x"][:])
    y_test = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    x_train = x_train.reshape(-1, x_train.shape[0])
    x_test = x_test.reshape(-1, x_test.shape[0])

    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, y_train, x_test, y_test, classes
