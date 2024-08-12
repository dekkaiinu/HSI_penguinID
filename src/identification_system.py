import numpy as np
import torch

from detect.hs2rgb.hs2rgb import hs2rgb
from detect.detector import detector

from identify.extract_pixels.extract_pixels import extract_pixels
from identify.pixel_wise_mlp.pixel_wise_mlp import pixel_wise_mlp
from identify.calc_penguin_id.calc_penguin_id import calc_penguin_id

def identification_system(hsi: np.ndarray, detect_model: torch.nn.Module, identify_model: torch.nn.Module, device: torch.device):
    rgb = hs2rgb(hsi)

    pred_bboxs = detector(rgb, device, detect_model, stride=detect_model.stride, conf_thres=0.45, iou_thres=0.25, classes=None, resize=640)

    hs_pixels = extract_pixels(hsi, pred_bboxs)

    predict_scores = []
    for hs_pixel in hs_pixels:
        hs_pixel = hs_pixel / 4095
        pred_score = pixel_wise_mlp(hs_pixel, identify_model, device)
        predict_scores.append(pred_score)

    preds, vote_rates = calc_penguin_id(predict_scores)

    return preds, pred_bboxs, vote_rates


if __name__ == "__main__":
    import h5py
    from detect.yolov5.models.common import DetectMultiBackend
    from identify.pixel_wise_mlp.models.mlp_batch_norm import MLP_BatchNorm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detect_model = DetectMultiBackend("/mnt/hdd1/youta/ws/HSI_penguinID/src/detect/yolov5/runs/train/exp3/weights/best.pt", device=device, dnn=False, data="yolov5/data/penguin_detection.yaml", fp16=True)
    
    identify_model = MLP_BatchNorm(input_dim=151, output_dim=16)
    identify_model.to(device)

    identify_model.load_state_dict(torch.load('/mnt/hdd1/youta/ws/HSI_penguinID/src/identify/pixel_wise_mlp/runs/2024-02-21/18-49/weight.pt'))

    # hdf5ファイルからhsデータを読み込む
    hdf5_path = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin/hyper_penguin.h5'
    with h5py.File(hdf5_path, 'r') as file:
        # 特定の画像IDを指定する必要がある．ここでは例として'20230623114016'を使用
        image_id = '20230627114441'
        hsi = file[f'hsi/{image_id}.npy'][:]
    print(hsi.shape)

    predict_id = identification_system(hsi=hsi, detect_model=detect_model, identify_model=identify_model, device=device)
    print(predict_id)