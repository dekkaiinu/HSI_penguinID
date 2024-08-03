import numpy as np
from tqdm import tqdm

def data_prepro(X_train, X_val, X_test, cfg_prepro):
    print("**************************************************")
    print("preprocessing")
    print("norm level:{}, method:{}".format(cfg_prepro.name, cfg_prepro.method))
    if cfg_prepro.name == "dataset":
        if cfg_prepro.method == "min-max":
            X_train_std = min_max(X_train, X_train, cfg_prepro.axis)
            X_val_std = min_max(X_val, X_train, cfg_prepro.axis)
            X_test_std = min_max(X_test, X_train, cfg_prepro.axis)
            if cfg_prepro.axis:
                print("axis=0:True")
            else:
                print("axis=0:False")
        elif cfg_prepro.method == "std":
            X_train_std = std(X_train, X_train, cfg_prepro.axis)
            X_val_std = std(X_val, X_train, cfg_prepro.axis)
            X_test_std = std(X_test, X_train, cfg_prepro.axis)
            if cfg_prepro.axis:
                print("axis=0:True")
            else:
                print("axis=0:False")
        elif cfg_prepro.method == "0-1":
            X_train_std = X_train / 4095
            X_val_std = X_val / 4095
            X_test_std = X_test / 4095

            # X_train_std = X_train
            # X_val_std = X_val
            # X_test_std = X_test
        elif cfg_prepro.method == "residual_img":
            X_train_std = residual_img(X_train, X_train)
            X_val_std = residual_img(X_val, X_train)
            X_test_std = residual_img(X_test, X_train)
        elif cfg_prepro.method == "iarr":
            X_train_std = iarr(X_train, X_train)
            X_val_std = iarr(X_val, X_train)
            X_test_std = iarr(X_test, X_train)
        else:
            X_train_std = X_train
            X_val_std = X_val
            X_test_std = X_test
    elif cfg_prepro.name == "pixwise":
        if cfg_prepro.method == "min-max":
            X_train_std = pix_wise_min_max(X_train)
            X_val_std = pix_wise_min_max(X_val)
            X_test_std = pix_wise_min_max(X_test)
        elif cfg_prepro.method == "std":
            X_train_std = pix_wise_std(X_train)
            X_val_std = pix_wise_std(X_val)
            X_test_std = pix_wise_std(X_test)
        elif cfg_prepro.method == "zero_wave":
            X_train_std = zero_wave(X_train)
            X_val_std = zero_wave(X_val)
            X_test_std = zero_wave(X_test)
        elif cfg_prepro.method == 'first_derivative':
            X_train_std = first_derivative4hs(X_train)
            X_val_std = first_derivative4hs(X_val)
            X_test_std = first_derivative4hs(X_test)
        elif cfg_prepro.method == 'second_derivative':
            X_train_std = second_derivative4hs(X_train)
            X_val_std = second_derivative4hs(X_val)
            X_test_std = second_derivative4hs(X_test)
    else:
        print("error")
    print("X_train.shape = {}, X_val.shape = {}, X_test.shape = {}".format(X_train_std.shape, X_val_std.shape, X_test_std.shape))
    return X_train_std, X_val_std, X_test_std

def min_max(X, X_std, axis):
    if axis:
        X = (X - np.min(X_std, axis=0)) / (np.max(X_std, axis=0) - np.min(X_std, axis=0))
    else:
        X = (X - np.min(X_std)) / (np.max(X_std) - np.min(X_std))
    return X

def std(X, X_std, axis):
    if axis:
        X = (X - np.mean(X_std, axis=0)) / np.std(X_std, axis=0)
    else:
        X = (X - np.mean(X_std)) / np.std(X_std)
    return X


def pix_wise_min_max(X):
    min_vals = X.min(axis=1, keepdims=True)
    max_vals = X.max(axis=1, keepdims=True)
    X_norm = (X - min_vals) / (max_vals - min_vals)
    return X_norm

def pix_wise_std(X):
    mean_vals = X.mean(axis=1, keepdims=True)
    std_vals = X.std(axis=1, keepdims=True)
    X_norm = (X - mean_vals) / std_vals
    return X_norm

def zero_wave(X):
    X = X / 4095
    # (データ数, 次元数)
    #引くデータのshapeは(データ数(各データごとに違う)，次元数(特定の波長が0になるように定数で引く))
    chosen_band = 60
    chosen_X = X[: , chosen_band]
    exted_chosen_X = np.repeat(chosen_X[:, np.newaxis], 151, axis=1)
    
    X_norm = X - exted_chosen_X
    return X_norm

def residual_img(X, X_train):
    # X_train = X_train / 4095
    # X = X / 4095

    # 選択したスペクトルの最大値を出す
    chosen_band = 60
    chosen_X = X_train[: , chosen_band]
    max_value = np.max(chosen_X)
    max_index = np.argmax(chosen_X)

    band_X = X[:, chosen_band]
    residual_band_X = max_value - band_X
    residual_X = X + np.repeat(residual_band_X[:, np.newaxis], 151, axis=1)

    X_norm = residual_X - np.mean(residual_X, axis=0)
    return X_norm

def iarr(X, X_train):
    # X_train = X_train / 4095
    # X = X / 4095

    X_norm = X / np.mean(X_train)

    return X_norm

def first_derivative4hs(X):
    X = X.astype(np.int32)
    X_first_derivative = []
    for index in range(X.shape[0]):
        x = X[index]
        x = x / 4096
        x_first_derivative = x[1:] - x[:len(x) - 1]
        X_first_derivative.append(x_first_derivative)
    X_first_derivative = np.array(X_first_derivative)
    return X_first_derivative

def second_derivative4hs(X):
    X = X.astype(np.int32)
    X_second_derivative = []
    for index in range(X.shape[0]):
        x = X[index]
        x = x / 4096
        x_first_derivative = x[1:] - x[:len(x) - 1]
        x_second_derivative = x_first_derivative[1:] - x_first_derivative[:len(x_first_derivative) - 1]
        X_second_derivative.append(x_second_derivative)
    X_second_derivative = np.array(X_second_derivative)
    return X_second_derivative