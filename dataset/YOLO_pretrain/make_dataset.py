import json
from tqdm import tqdm
import h5py
import cv2
import numpy as np

def make_dataset(dataset_dir, dataset_info, save_dir, flip=False):
    target_id_list = ['0373', '0143', '0346', '0166', '0566', '0126', '0473', '0456', '0146', '0356', '0363', '0133', '0553', '0376', '0343', '0477']
    hdf5_path = f'{dataset_dir}/hyper_penguin.h5'
    h5_file = h5py.File(hdf5_path, 'r')
    for data_info in tqdm(dataset_info):
        # img = (h5_file[f'rgb/{data_info["image_id"]}.png'][:]).astype('uint8')
        img = (h5_file[f'rgb/{data_info["image_id"]}'][:]).astype('uint8')

        if flip:
            img = cv2.flip(img, 1)

        for annotation in data_info['annotation']:
            if annotation['penguin_id'] not in target_id_list:
                continue
            bbox = annotation['bbox']
            if flip:
                bbox = [2048 - bbox[0] - bbox[2], bbox[1], bbox[2], bbox[3]]
            penguin_id = target_id_list.index(annotation['penguin_id'])
            text = yolo_format_data(bbox, penguin_id)
            with open(f'{save_dir}/{data_info["image_id"]}.txt', "a") as txt_file:
                txt_file.write(text + '\n')
            
        img_path = f'{save_dir}/{data_info["image_id"]}.png'
        cv2.imwrite(img_path, img)
    h5_file.close()


def yolo_format_data(bbox, target_id, width=2048, height=1080):
    center_x = bbox[0] + (bbox[2] / 2)
    center_y = bbox[1] + (bbox[3] / 2)
    text = (str(target_id) + ' ' + str(round(center_x / width, 4)) + ' ' + str(round(center_y / height, 4)) + ' ' + 
            str(round(bbox[2] / width, 4)) + ' ' + str(round(bbox[3] / height, 4)))
    return text



if __name__ == '__main__':
    # dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin'
    # save_dir = './dataset'

    # dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', flip=True)
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', flip=True)
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', flip=True)

    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin_4'
    save_dir = './dataset_4'

    # dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', flip=True)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', flip=False)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', flip=False)

    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin_8'
    save_dir = './dataset_8'

    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', flip=True)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', flip=False)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', flip=False)

    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin_16'
    save_dir = './dataset_16'

    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    # make_dataset(dataset_dir, dataset_info, f'{save_dir}/train', flip=True)
    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/val', flip=False)
    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test', flip=False)