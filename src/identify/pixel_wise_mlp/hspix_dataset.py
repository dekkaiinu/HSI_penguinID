import csv
import numpy as np
import torch
import torch.utils.data as data

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
        '''
        '''
        hs_pixel, target = self.feature[index], self.target[index]

        hs_pixel = torch.tensor(hs_pixel, dtype=torch.float32) / 4095
        target = torch.tensor(target, dtype=torch.long)
    
        return hs_pixel, target

    def __len__(self):
        return len(self.target)