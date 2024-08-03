import numpy as np
import torch

from detect.hs2rgb.hs2rgb import hs2rgb
from detect.detector import detector

from identify.extract_pixels.extract_pixels import extract_pixels
from identify.pixel_wise_mlp.pixel_wise_mlp import pixel_wise_mlp
from identify.calc_penguin_id.calc_penguin_id import calc_penguin_id

def main(hsi: np.ndarray, detect_model: torch.nn.Module, identify_model: torch.nn.Module, device: torch.device):
    rgb = hs2rgb(hsi)

    pred_bboxs = detector(rgb, device, detect_model, stride=model.stride, conf_thres=0.45, iou_thres=0.25, classes=None, resize=640)

    hs_pixels = extract_pixels(hsi, pred_bboxs)

    predict_scores = []
    for hs_pixel in hs_pixels:
        hs_pixel = hs_pixel / 4095
        pred_score = pixel_wise_mlp(hs_pixel, identify_model, device)
        predict_scores.append(pred_score)

    preds = calc_penguin_id(predict_scores)

    return preds


if __name__ == "__main__":
    from detect.yolov5.models.common import DetectMultiBackend
    from identify.pixel_wise_mlp.models.MLP_BatchNorm import MLP_BatchNorm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detect_model = DetectMultiBackend("yolov5/weights/penguin_detection.pt", device=device, dnn=False, data="yolov5/data/penguin_detection.yaml", fp16=True)
    
    identify_model = MLP_BatchNorm(input_dim=151, output_dim=16)
    identify_model.to(device)

    identify_model.load_state_dict(torch.load('/mnt/hdd1/youta/ws/HSI_penguinID/src/identify/pixel_wise_mlp/runs/2024-02-21/18-49/weight.pt'))
    main(hsi=hsi, detect_model=detect_model, identify_model=identify_model, device=device)