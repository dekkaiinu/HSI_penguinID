import numpy as np

def noise_ref(X_list, pix_num=40):
    refed_X = []
    for X in X_list:
        new_shape = (pix_num, X.shape[0] // pix_num, X.shape[1])
        reshaped_X = X[:new_shape[1] * pix_num].reshape(new_shape)
        mean_X = np.mean(reshaped_X,axis=0)
        refed_X.append(mean_X)

    return refed_X
