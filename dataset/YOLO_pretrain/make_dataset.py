import json
from tqdm import tqdm
import h5py
import cv2

def make_dataset(dataset_dir, dataset_info, save_dir):
    hdf5_path = f'{dataset_dir}/hyper_penguin.hdf5'
    file = h5py.File(hdf5_path, 'r')
    for data_info in tqdm(dataset_info):
        img = file[f'rgb/{data_info["image_id"]}.png'][:]
        
        for annotation in data_info['annotation']:
            text = yolo_format_data(annotation['bbox'], 0, img.shape[1], img.shape[0])
            with open(f'{save_dir}/{data_info["image_id"]}.txt', "a") as file:
                file.write(text + '\n')
            
            img_path = f'{save_dir}/{data_info["image_id"]}.png'
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img_bgr)

def yolo_format_data(bbox, target_id, width=2048, height=1080):
    center_x = bbox[0] + (bbox[2] / 2)
    center_y = bbox[1] + (bbox[3] / 2)
    text = (str(target_id) + ' ' + str(round(center_x / width, 4)) + ' ' + str(round(center_y / height, 4)) + ' ' + 
            str(round(bbox[2] / width, 4)) + ' ' + str(round(bbox[3] / height, 4)))
    return text

if __name__ == '__main__':
    dataset_dir = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin'
    save_dir = './dataset'

    dataset_info = json.load(open('../train_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/train')

    dataset_info = json.load(open('../validation_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/validation')

    dataset_info = json.load(open('../test_dataset_info.json', 'r'))
    make_dataset(dataset_dir, dataset_info, f'{save_dir}/test')