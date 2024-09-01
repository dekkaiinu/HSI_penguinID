import numpy as np
from sklearn.mixture import GaussianMixture

def weight_mask(crop_patch: np.ndarray):
    flattened_patch = crop_patch.reshape(-1, crop_patch.shape[-1])

    weights = cluster_weight(flattened_patch)
    
    weight_mask = weights.reshape(crop_patch.shape[:-1])

    return weight_mask

def cluster_weight(spectrum):
    normalized_spectrum = (spectrum - np.min(spectrum, axis=0)) / (np.max(spectrum, axis=0) - np.min(spectrum, axis=0))
    gmm = GaussianMixture(n_components=2)
    gmm.fit(normalized_spectrum)
    labels = gmm.predict(normalized_spectrum)

    average_0 = np.average(spectrum[labels == 0])
    average_1 = np.average(spectrum[labels == 1])

    if average_0 > average_1:
        labels = np.where(labels == 0, 1, 0)
    return labels