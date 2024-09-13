import json
import h5py
import numpy as np
from tqdm import tqdm
import os
import csv
from sklearn.mixture import GaussianMixture

from hsitools.correction import hsi_blur, hsi_gaussian_blur


def make_dataset(dataset_dir, dataset_info, save_dir, resize_n=1, place = 'all', crop_ratio=0.2, blur_size=1):
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
            if resize_n != 1:
                hsi = file[f'hsi/{data_info["image_id"]}'][:]
            else:
                hsi = file[f'hsi/{data_info["image_id"]}.npy'][:]
            
            if blur_size > 1:
                hsi = hsi_blur(hsi, blur_size)
                # hsi = hsi_gaussian_blur(hsi, blur_size)
            for annotation in data_info['annotation']:
                penguin_id = annotation['penguin_id']
                if penguin_id not in target_id_list:
                    continue

                bbox = annotation['bbox']
                bbox = [int(bbox[0] // resize_n), int(bbox[1] // resize_n), int((bbox[0] + bbox[2]) // resize_n), int((bbox[1] + bbox[3]) // resize_n)]
                # croped_patch = crop_patch(hsi, bbox, crop_size=crop_size)
                croped_patch = crop_patch_ratio(hsi[bbox[1]:bbox[3], bbox[0]:bbox[2]], crop_ratio=crop_ratio)

                if place == 'all':
                    hs_pixels = croped_patch.reshape(-1, croped_patch.shape[-1])
                elif place == 'black':
                    weight_mask = weight_patch(croped_patch)
                    hs_pixels = croped_patch[weight_mask == 0]
                elif place == 'white':
                    weight_mask = weight_patch(croped_patch)
                    hs_pixels = croped_patch[weight_mask == 1]
                
                if len(hs_pixels) > 2000:
                    np.random.seed(42)
                    indices = np.random.choice(len(hs_pixels), 1000, replace=False)
                    hs_pixels = hs_pixels[indices]
                
                data_writer.writerows(hs_pixels)
                penguin_id = target_id_list.index(penguin_id)
                target_writer.writerows([[penguin_id]] * len(hs_pixels))

def crop_patch_ratio(hsi: np.ndarray, crop_ratio=0.2) -> np.ndarray:
    x_center = int(hsi.shape[1] * 0.5)
    y_center = int(hsi.shape[0] * 0.5)

    left = x_center - int((hsi.shape[1]) * crop_ratio / 2)
    right = x_center + int((hsi.shape[1]) * crop_ratio / 2)
    top = y_center - int((hsi.shape[0]) * crop_ratio / 2)
    bottom = y_center + int((hsi.shape[0]) * crop_ratio / 2)
    crop = hsi[top:bottom, left:right]
    return crop

def crop_patch(hsi: np.ndarray, bboxs: list, crop_size: int = 64) -> np.ndarray:
    """
    Crop a patch from a hyperspectral image based on bounding box coordinates.

    Args:
        hsi (np.ndarray): Hyperspectral image array with shape (height, width, channels).
        bboxs (list): Bounding box coordinates [x1, y1, x2, y2] normalized to [0, 1].
        crop_size (int, optional): Size of the square crop. Defaults to 64.

    Returns:
        np.ndarray: Cropped patch with shape (crop_size, crop_size, channels).
                    If the crop extends beyond the image boundaries, the result is padded with zeros.
    """
    center_x = int((bboxs[0] + bboxs[2]) / 2)
    center_y = int((bboxs[1] + bboxs[3]) / 2)

    start_x = max(0, center_x - crop_size // 2)
    end_x = min(hsi.shape[1], center_x + crop_size // 2)
    start_y = max(0, center_y - crop_size // 2)
    end_y = min(hsi.shape[0], center_y + crop_size // 2)
    
    cropped_hsi = hsi[start_y:end_y, start_x:end_x, :]
    
    if cropped_hsi.shape[0] < crop_size or cropped_hsi.shape[1] < crop_size:
        padded_hsi = np.zeros((crop_size, crop_size, hsi.shape[2]), dtype=hsi.dtype)
        padded_hsi[:cropped_hsi.shape[0], :cropped_hsi.shape[1], :] = cropped_hsi
        cropped_hsi = padded_hsi
    
    return cropped_hsi

def weight_patch(crop_patch: np.ndarray):
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

if __name__ == '__main__':
    # dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin_16'
    # save_dir = './all_16'
    # dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=16, place='all', crop_ratio=0.35, blur_size=1)
    # dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=16, place='all', crop_ratio=0.35, blur_size=1)
    # dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=16, place='all', crop_ratio=0.35, blur_size=1)
    # save_dir = './white_16'
    # dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=16, place='white', crop_ratio=0.35, blur_size=1)
    # dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=16, place='white', crop_ratio=0.35, blur_size=1)
    # dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=16, place='white', crop_ratio=0.35, blur_size=1)
    # save_dir = './black_16'
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=16, place='black', crop_ratio=0.35, blur_size=1)
    # dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=16, place='black', crop_ratio=0.35, blur_size=1)
    # dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=16, place='black', crop_ratio=0.35, blur_size=1)



    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin_8'
    save_dir = './blur_5/all_8'
    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=8, place='all', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=8, place='all', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=8, place='all', crop_ratio=0.35, blur_size=3)

    save_dir = './blur_5/white_8'
    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=8, place='white', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=8, place='white', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=8, place='white', crop_ratio=0.35, blur_size=3)

    save_dir = './blur_5/black_8'
    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=8, place='black', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=8, place='black', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=8, place='black', crop_ratio=0.35, blur_size=3)



    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin_4'
    save_dir = './blur_5/all_4'
    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=4, place='all', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=4, place='all', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=4, place='all', crop_ratio=0.35, blur_size=3)

    save_dir = './blur_5/white_4'
    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=4, place='white', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=4, place='white', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=4, place='white', crop_ratio=0.35, blur_size=3)

    save_dir = './blur_5/black_4'
    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=4, place='black', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=4, place='black', crop_ratio=0.35, blur_size=3)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=4, place='black', crop_ratio=0.35, blur_size=3)



    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin'
    # save_dir = './blur_5/all'
    # dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=1, place='all', crop_ratio=0.35, blur_size=5)
    # dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=1, place='all', crop_ratio=0.35, blur_size=5)
    # dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=1, place='all', crop_ratio=0.35, blur_size=5)

    save_dir = './blur_5/white'
    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=1, place='white', crop_ratio=0.35, blur_size=5)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=1, place='white', crop_ratio=0.35, blur_size=5)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=1, place='white', crop_ratio=0.35, blur_size=5)

    save_dir = './blur_5/black'
    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', resize_n=1, place='black', crop_ratio=0.35, blur_size=5)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', resize_n=1, place='black', crop_ratio=0.35, blur_size=5)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', resize_n=1, place='black', crop_ratio=0.35, blur_size=5)



