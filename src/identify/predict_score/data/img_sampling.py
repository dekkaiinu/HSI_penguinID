import numpy as np

def img_sampling(X_train, X_val, X_test, y_train, y_val, y_test, cfg_sampling):
    print("**************************************************")
    print("sampling method: {}".format(cfg_sampling.method))
    print("train image size:{}x{}, val and test image size:{}x{}".format(cfg_sampling.train, len(set(y_train)),
                                                                          cfg_sampling.val_test, len(set(y_val))))
    X_train, y_train = sampling_data(X_train, y_train, cfg_sampling.train)
    X_val, y_val = sampling_data(X_val, y_val, cfg_sampling.val_test)
    X_test, y_test = sampling_data(X_test, y_test, cfg_sampling.val_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def sampling_data(features, labels, target_count):
    unique_labels = set(labels)
    sampled_features = []
    sampled_labels = []

    for label in unique_labels:
        label_indices = [i for i, x in enumerate(labels) if x == label]
        if len(label_indices) < target_count:
            raise ValueError(f"The number of samples with label {label} is less than target_count {target_count}")
        
        selected_indices = np.random.choice(label_indices, size=target_count, replace=False)
        for index in selected_indices:
            sampled_features.append(features[index])
            sampled_labels.append(label)

    return sampled_features, sampled_labels