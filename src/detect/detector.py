import torch
import numpy as np
import cv2

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression


def detector(img, device, model, stride, conf_thres, iou_thres, classes, resize):
    h, w, _ = img.shape
    # Padded resize
    img = letterbox(img, resize, stride=stride, auto=True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=50)
    pred_cpu = pred[0].cpu().detach().numpy()

    pred_bboxs = []
    for p in pred_cpu:
        # width norm
        p[0] = p[0] / img.shape[2]
        p[2] = p[2] / img.shape[2]
        # height norm
        p[1] = p[1] / img.shape[1]
        p[3] = p[3] / img.shape[1]
        # drop class label
        p = np.delete(p, [4, 5])
        # p = [x1, y1, x2, y2]
        # append list
        pred_bboxs.append(p)

    return pred_bboxs