import numpy as np

def pix_sampling(X_train, X_val, X_test, y_train, y_val, y_test, cfg_sampling):
    print("**************************************************")
    print("sampling method: {}".format(cfg_sampling.method))
    print("train data size:{}x{}, val and test data size:{}x{}".format(cfg_sampling.train, len(np.unique(y_train)),
                                                                        cfg_sampling.val_test, len(np.unique(y_test))))
    X_train, y_train = sampling_data(X_train, y_train, cfg_sampling.train)
    X_val, y_val = sampling_data(X_val, y_val, cfg_sampling.val_test)
    X_test, y_test = sampling_data(X_test, y_test, cfg_sampling.val_test)
    return X_train, X_val, X_test, y_train, y_val, y_test

def sampling_data(features, labels, target_count):
    unique_labels = np.unique(labels)
    balanced_features = []
    balanced_labels = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.seed(0)
        selected_indices = np.random.choice(label_indices, size=target_count, replace=False)
        balanced_features.extend(features[selected_indices])
        balanced_labels.extend([label] * target_count)

    return np.array(balanced_features), np.array(balanced_labels)