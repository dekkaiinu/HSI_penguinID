import numpy as np
from tqdm import tqdm

def list2array(X_list, y_list):
    if X_list[0].ndim > 1:
        # X = np.concatenate(X_list, axis=0)
        X = np.empty((0, X_list[0].shape[1]))
        batch_size = 2000
        for i in tqdm(range(0, len(X_list), batch_size)):
            X_batch = np.concatenate(X_list[i:i + batch_size], axis=0)
            X = np.concatenate([X, X_batch], axis=0)
    else:
        X = np.array(X_list)
    if X.shape[0] == len(y_list):
        y = np.array(y_list)
    else:
        y = np.array(label_maker(X_list, y_list))
    return X, y

def label_maker(X_list, y_list):
    y_array = []
    for x, y in zip(X_list, y_list):
        for i in range(x.shape[0]):
            y_array.append(y)
    return y_array