import csv
import numpy as np
import torch
import torch.utils.data as data

from collections import Counter
import random


class HsPixelDataset(data.Dataset):
    '''
    HsPixelDataset is a custom dataset class for handling hyperspectral pixel data.
    
    Attributes:
        feature (Any): The input features for the dataset.
        target (Any): The target labels for the dataset.
    '''
    def __init__(self, feature_path, target_path):
        super().__init__()
        self.feature = np.loadtxt(feature_path, delimiter=',')
        self.target = np.loadtxt(target_path, delimiter=',', dtype=int)


    def __getitem__(self, index: int):
        hs_pixel, target = self.feature[index].reshape(-1, 1, 1), self.target[index]

        hs_pixel = torch.tensor(hs_pixel, dtype=torch.float32) / 4095
        target = torch.tensor(target, dtype=torch.long)

        return hs_pixel, target

    def __len__(self):
        return len(self.target)


    def balance_classes(self):
        class_counts = Counter(self.target)
        min_class_count = min(class_counts.values())
        max_class_count = min_class_count * 2

        print("Before balancing:", class_counts)

        balanced_features = []
        balanced_targets = []

        for cls in class_counts:
            cls_indices = [i for i, t in enumerate(self.target) if t == cls]
            if len(cls_indices) > max_class_count:
                cls_indices = random.sample(cls_indices, max_class_count)
            balanced_features.extend(self.feature[cls_indices])
            balanced_targets.extend(self.target[cls_indices])

        self.feature, self.target = np.array(balanced_features), np.array(balanced_targets)

        balanced_class_counts = Counter(self.target)
        print("After balancing:", balanced_class_counts)