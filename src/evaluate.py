import os
import json
import h5py
import torch
from tqdm import tqdm

from detect.yolov5.models.common import DetectMultiBackend
from identify.pixel_wise_mlp.models.mlp_batch_norm import MLP_BatchNorm
from identification_system import identification_system

def evaluate(data_info, dataset_path, detect_path, identify_path, save_path, device):
    detect_model = DetectMultiBackend(detect_path, device=device, dnn=False, data="yolov5/data/penguin_detection.yaml", fp16=True)
    
    identify_model = MLP_BatchNorm(input_dim=151, output_dim=16)
    identify_model.to(device)
    identify_model.load_state_dict(torch.load(identify_path))
    
    pred_path = os.path.join(save_path, 'pred')
    gt_path = os.path.join(save_path, 'gt')    
    os.makedirs(pred_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    for data in tqdm(data_info):
        image_id = data['image_id']
        hsi = h5py.File(dataset_path, 'r')[f'hsi/{image_id}.npy'][:]
        preds, pred_bboxs, vote_rates = identification_system(hsi=hsi, detect_model=detect_model, identify_model=identify_model, device=device)
        write_detect_format_data(preds, pred_bboxs, vote_rates, os.path.join(pred_path, f'{image_id}.txt'))
        write_gt_format_data(data['annotation'], os.path.join(gt_path, f'{image_id}.txt'))
    return None


def write_detect_format_data(pred_ids, pred_bboxs, confidences, save_path, width=2048, height=1080):
    for pred_id, pred_bbox, confidence in zip(pred_ids, pred_bboxs, confidences):
        text = (str(pred_id) + " " + str(round(confidence, 4)) 
                + " " + str(int(pred_bbox[0] * width)) + " " + str(int(pred_bbox[1] * height))
                + " " + str(int(pred_bbox[2] * width)) + " " + str(int(pred_bbox[3] * height)))
        with open(save_path, "a") as file:
                file.write(text + "\n")

def write_gt_format_data(annotation, save_path, width=2048, height=1080):
    target_id_list = ['0373', '0143', '0346', '0166', '0566', '0126', '0473', '0456', '0146', '0356', '0363', '0133', '0553', '0376', '0343', '0477']
    for ann in annotation:
        if ann['penguin_id'] == '0000':
            continue
        id = target_id_list.index(ann['penguin_id'])
        text = (str(id)
                + " " + str(int(ann['bbox'][0] * width)) + " " + str(int(ann['bbox'][1] * height))
                + " " + str(int(ann['bbox'][2] * width)) + " " + str(int(ann['bbox'][3] * height)))
        with open(save_path, "a") as file:
                file.write(text + "\n")

if __name__ == "__main__":
    detect_path = "/mnt/hdd1/youta/ws/HSI_penguinID/src/detect/yolov5/runs/train/exp3/weights/best.pt"
    identify_path = "/mnt/hdd1/youta/ws/HSI_penguinID/src/identify/pixel_wise_mlp/runs/2024-02-21/18-49/weight.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path ='/mnt/hdd3/datasets/hyper_penguin/hyper_penguin/hyper_penguin.h5'
    with open('/mnt/hdd1/youta/ws/HSI_penguinID/dataset/test_dataset_info.json', 'r') as f:
        data_info = json.load(f)
    save_path = '/mnt/hdd1/youta/ws/HSI_penguinID/src/evaluate/result'

    evaluate(data_info, dataset_path, detect_path, identify_path, save_path, device)