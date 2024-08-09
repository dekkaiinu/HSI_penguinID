import json
import h5py
import numpy as np
from tqdm import tqdm
import os
import csv

from hsitools.correction.blur import hsi_gaussian_blur
from hsitools.convert import extract_pixels_from_hsi_mask


def make_dataset(dataset_dir, dataset_info, save_dir):
    target_id_list = ['0373', '0143', '0346', '0166', '0566', '0126', '0473', '0456', '0146', '0356', '0363', '0133', '0553', '0376', '0343', '0477']
    hdf5_path = f'{dataset_dir}/hyper_penguin.h5'
    output_data_path = f'{save_dir}_data.csv'
    output_target_path = f'{save_dir}_target.csv'
    
    with h5py.File(hdf5_path, 'r') as file, \
         open(output_data_path, 'w', newline='') as data_file, \
         open(output_target_path, 'w', newline='') as target_file:
        
        data_writer = csv.writer(data_file)
        target_writer = csv.writer(target_file)
        
        for data_info in tqdm(dataset_info):
            hsi = file[f'hsi/{data_info["image_id"]}.npy'][:]
            hsi = hsi_gaussian_blur(hsi, kernel_size=5, sigmaX=1)
            for annotation in data_info['annotation']:
                penguin_id = annotation['penguin_id']
                if penguin_id not in target_id_list:
                    continue

                mask = file[f'seg_mask/{annotation["segmentation_mask"]}'][:]
                hs_pixels = extract_pixels_from_hsi_mask(hsi, mask)
                
                data_writer.writerows(hs_pixels)
                penguin_id = target_id_list.index(penguin_id)
                target_writer.writerows([[penguin_id]] * len(hs_pixels))


if __name__ == '__main__':
    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin'
    save_dir = '.'

    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train')

    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/validation')
    
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test')