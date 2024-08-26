import os
import cv2
import json
import h5py
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from detect.yolov5.models.common import DetectMultiBackend
from identify.pixel_wise_mlp.models.mlp_batch_norm import MLP_BatchNorm
from identification_system import identification_system
from plot_rectangle import plot_rectangle

def evaluate(data_info, dataset_path, detect_path, detect_yaml, identify_path, save_path, device):
    detect_model = DetectMultiBackend(detect_path, device=device, dnn=False, data=detect_yaml, fp16=True)
    
    identify_model = MLP_BatchNorm(input_dim=151, output_dim=16)
    identify_model.to(device)
    identify_model.load_state_dict(torch.load(identify_path))
    
    # pred_path = os.path.join(save_path, 'pred')
    # gt_path = os.path.join(save_path, 'gt')    
    # os.makedirs(pred_path, exist_ok=True)
    # os.makedirs(gt_path, exist_ok=True)
    predictions = defaultdict(list)
    ground_truths = defaultdict(list)

    for data in tqdm(data_info):
        height = data['meta_data']['height']
        width = data['meta_data']['width']
        target_id_list = ['0373', '0143', '0346', '0166', '0566', '0126', '0473', '0456', '0146', '0356', '0363', '0133', '0553', '0376', '0343', '0477']
        flag = False
        gt_ids = []
        gt_bboxs = []
        for ann in data['annotation']:
            if ann['penguin_id'] == '0000':
                continue
            id = target_id_list.index(ann['penguin_id'])
            gt_bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
            ground_truths[id].append(gt_bbox)

            gt_bbox = [ann['bbox'][0] / width, ann['bbox'][1] / height, (ann['bbox'][0] + ann['bbox'][2]) / width, (ann['bbox'][1] + ann['bbox'][3]) / height]
            gt_ids.append(id)
            gt_bboxs.append(gt_bbox)
            flag = True
        if not flag:
            continue
        
        image_id = data['image_id']
        hsi = h5py.File(dataset_path, 'r')[f'hsi/{image_id}.npy'][:]
        rgb = h5py.File(dataset_path, 'r')[f'rgb/{image_id}.png'][:]
        preds, pred_bboxs, vote_rates = identification_system(hsi=hsi, detect_model=detect_model, identify_model=identify_model, device=device, rgb=rgb)
  
        # write_detect_format_data(preds, pred_bboxs, vote_rates, os.path.join(pred_path, f'{image_id}.txt'))
        # write_gt_format_data(data['annotation'], os.path.join(gt_path, f'{image_id}.txt'))
        for pred_id, pred_bbox, confidence in zip(preds, pred_bboxs, vote_rates):
            pred_bbox = [confidence, int(pred_bbox[0] * hsi.shape[1]), int(pred_bbox[1] * hsi.shape[0]), int(pred_bbox[2] * hsi.shape[1]), int(pred_bbox[3] * hsi.shape[0])]
            predictions[pred_id].append(pred_bbox)
        pred_img = plot_rectangle(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR), pred_bboxs, preds, border_color='cyan', label_size=13, line_width=2)
        tar_img = plot_rectangle(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR), gt_bboxs, gt_ids, border_color='red', label_size=13, line_width=2)
        cv2.imwrite(os.path.join(save_path, f'{image_id}_pred.png'), pred_img)
        cv2.imwrite(os.path.join(save_path, f'{image_id}_tar.png'), tar_img)
    
    # 評価の実行
    mAP, aps = calculate_ap(predictions, ground_truths, len(target_id_list))

    print(f'mAP: {mAP:.4f}')
    result = {
        'mAP': mAP,
        'APs': {f'class_{i}': ap for i, ap in enumerate(aps)}
    }
    
    with open(os.path.join(save_path, 'evaluation_result.json'), 'w') as f:
        json.dump(result, f, indent=4)
    print('saved')
    return mAP

def calculate_ap(predictions, ground_truths, class_num):
    aps = []
    for class_id in range(class_num):
        if class_id not in ground_truths or class_id not in predictions:
            aps.append(0)
            continue
        
        pred = np.array(predictions[class_id])
        gt = np.array(ground_truths[class_id])
        print(pred)
        print(gt)
        if len(pred) == 0 or len(gt) == 0:
            aps.append(0)
            continue
        
        # 信頼度でソート
        pred = pred[pred[:, 0].argsort()[::-1]]

        tp = np.zeros(len(pred))
        fp = np.zeros(len(pred))
        gt_matched = np.zeros(len(gt), dtype=bool)
        
        for i, p in enumerate(pred):
            overlaps = calculate_iou(p[1:].reshape(1, -1), gt)
            max_iou = np.max(overlaps)
            max_idx = np.argmax(overlaps)
            
            if max_iou >= 0.5:
                if not gt_matched[max_idx]:
                    tp[i] = 1
                    gt_matched[max_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        # 累積和の計算
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / len(gt)
        prec = cum_tp / (cum_tp + cum_fp)
        
        # APの計算
        ap = compute_ap(rec, prec)
        aps.append(ap)
    
    return np.mean(aps), aps

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calculate_iou(boxes1, boxes2):
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1[:, None] + area2 - intersection

    iou = intersection / union
    return iou

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
    detect_path = "detect/yolov5/runs/train/exp5/weights/best.pt"
    detect_yaml = "/mnt/hdd1/youta/ws/HSI_penguinID/dataset/YOLO_pretrain/dataset/penguin_id_yolo.yaml"
    identify_path = '/mnt/hdd1/youta/ws/HSI_penguinID/src/identify/pixel_wise_mlp/runs/2024-08-15/13-22/weight.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path ='/mnt/hdd3/datasets/hyper_penguin/hyper_penguin/hyper_penguin.h5'
    with open('/mnt/hdd1/youta/ws/HSI_penguinID/dataset/test_dataset_info.json', 'r') as f:
        data_info = json.load(f)
    save_path = './result/'

    evaluate(data_info, dataset_path, detect_path, detect_yaml, identify_path, save_path, device)