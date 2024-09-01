import numpy as np
from data.list2array import list2array

def dataset_images2array(X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list):
    print("**************************************************")
    print("image to pix-array")
    X_train, y_train = list2array(X_train_list, y_train_list)
    X_val, y_val = list2array(X_val_list, y_val_list)
    X_test, y_test = list2array(X_test_list, y_test_list)

    # X_train, y_train = mean(X_train_list, y_train_list)
    # X_val, y_val = mean(X_val_list, y_val_list)
    # X_test, y_test = mean(X_test_list, y_test_list)

    print("train_data.shape = {}, val_data.shape = {}, test_data.shape = {}".format(X_train.shape, X_val.shape, X_test.shape))
    return X_train, X_val, X_test, y_train, y_val, y_test

def mean(X_list, y_list):
    X_mean = []
    for X in X_list:
        x_mean = np.mean(X, axis=0)
        X_mean.append(x_mean)
    X_mean = np.array(X_mean)
    y_list = np.array(y_list)
    return X_mean, y_list