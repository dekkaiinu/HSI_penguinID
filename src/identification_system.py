import numpy as np
import torch

from detect.hs2rgb.hs2rgb import hs2rgb
from detect.detector import detector

from identify.crop_patch.crop_patch import crop_patch
from identify.weight_mask.weight_mask import weight_mask
from identify.predict_score.predict_score import GetScoreModule
from identify.calc_penguin_id.calc_penguin_id import calc_penguin_id, argmax_id


def identification_system(hsi: np.ndarray, detect_model: torch.nn.Module, identify_model: torch.nn.Module, device: torch.device, rgb: np.ndarray = None, saliency_map: str = 'all', id_resolber: bool = True):
    if rgb is None:
        rgb = hs2rgb(hsi)
    else:
        rgb = rgb

    pred_bboxs = detector(rgb, device, detect_model, stride=detect_model.stride, conf_thres=0.45, iou_thres=0.25, classes=None, resize=640)

    get_score_module = GetScoreModule(identify_model)
    get_score_module.to(device)
    get_score_module.eval()

    preds = []
    for pred_bbox in pred_bboxs:
        croped_patch = crop_patch(hsi, pred_bbox, crop_size=64)

        if saliency_map == 'all':
            mask = np.ones((croped_patch.shape[0], croped_patch.shape[1]))
        elif saliency_map == 'black':
            mask = weight_mask(croped_patch)
            mask = 1 - mask
        elif saliency_map == 'white':
            mask = weight_mask(croped_patch)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        mask = mask.to(device)

        # import cv2
        # import numpy as np
        # from hsitools.convert import hs_to_rgb, gamma_correction
        # mask = mask * 255
        # cv2.imwrite('mask.png', mask.squeeze(0).cpu().numpy())
        # exit()
        input = (torch.tensor(croped_patch, dtype=torch.float32) / 4095).permute(2, 0, 1).unsqueeze(0)
        input = input.to(device)

        pred = get_score_module(input, mask)
        pred = pred.data.cpu().numpy().squeeze()
        preds.append(pred)
        # print(pred)
    
    if id_resolber:
        pred_ids, scores = calc_penguin_id(preds)
    else:
        pred_ids, scores = argmax_id(preds)
    # print('---------------------')
    # print(pred_ids)
    # print(scores)
    # print('---------------------')

    return pred_ids, pred_bboxs, scores


if __name__ == "__main__":
    import h5py
    from detect.yolov5.models.common import DetectMultiBackend
    from identify.predict_score.models.pix_wise_cnn import PointWiseCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detect_model = DetectMultiBackend("detect/yolov5/runs/train/exp5/weights/best.pt", device=device, dnn=False, data="/mnt/hdd1/youta/ws/HSI_penguinID/dataset/YOLO_pretrain/dataset/penguin_id_yolo.yaml", fp16=True)
    
    identify_model = PointWiseCNN(input_channels=151, output_channels=16, dropout_prob=0.5)
    identify_model.to(device)

    identify_model.load_state_dict(torch.load('/mnt/hdd1/youta/ws/HSI_penguinID/src/identify/predict_score/runs/2024-08-29/full_white/weight.pt'))

    # hdf5ファイルからhsデータを読み込む
    hdf5_path = '/mnt/hdd3/datasets/hyper_penguin/hyper_penguin/hyper_penguin.h5'
    with h5py.File(hdf5_path, 'r') as file:
        # 特定の画像IDを指定する必要がある．ここでは例として'20230623114016'を使用
        image_id = '20230623104020'
        hsi = file[f'hsi/{image_id}.npy'][:]
    print(hsi.shape)

    predict_ids, pred_bboxs, scores = identification_system(hsi=hsi, detect_model=detect_model, identify_model=identify_model, device=device, saliency_map='white', id_resolber=True)
    print('predict_ids:', predict_ids)
    print('pred_bboxs:', pred_bboxs)
    print('scores:', scores)