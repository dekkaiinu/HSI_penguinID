import json
import h5py
import numpy as np
from tqdm import tqdm

from hsitools.correction.blur import hsi_gaussian_blur
from hsitools.convert import extract_pixels_from_hsi_mask

def make_dataset(dataset_dir, dataset_info, save_dir):
    hdf5_path = f'{dataset_dir}/hyper_penguin.hdf5'
    file = h5py.File(hdf5_path, 'r')
    data = []
    target = []
    for data_info in tqdm(dataset_info):
        hsi = file[f'hsi/{data_info["image_id"]}.png'][:]
        for annotation in data_info['annotation']:
            mask = file[f'mask/{annotation["segmentation_mask"]}.png'][:]
            penguin_id = annotation['penguin_id']

            hsi = hsi_gaussian_blur(hsi, kernel_size=5, sigmaX=1)
            hs_pixels = extract_pixels_from_hsi_mask(hsi, mask)

            data.append(hs_pixels)
            target.append(penguin_id)    
    data = np.array(data)
    target = np.array(target)
    np.savez(f'{save_dir}.npz', data=data, target=target)


if __name__ == '__main__':
    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin'
    save_dir = '.'

    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train')

    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/validation')
    
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test')