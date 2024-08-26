import numpy as np
import cv2

def extract_pixels(hsi: np.ndarray, bboxs: list):
    width = hsi.shape[1]
    height = hsi.shape[0]
    hs_pixels = []
    for bbox in bboxs:
        mask = bbox_center_mask(bbox, width, height)
        hs_pixels.append(hsi[mask == 255])
    return hs_pixels

def bbox_center_mask(bbox: np.ndarray, img_width, img_height, crop_rate=0.2):
    x1 = int(bbox[0] * img_width)
    x2 = int(bbox[2] * img_width)
    y1 = int(bbox[1] * img_height)
    y2 = int(bbox[3] * img_height)

    x_center = int((x1 + x2) * 0.5)
    y_center = int((y1 + y2) * 0.5)

    # [left, top, right, bottom]
    left = x_center - int((x2 - x1) * crop_rate / 2)
    right = x_center + int((x2 - x1) * crop_rate / 2)
    top = y_center - int((y2 - y1) * crop_rate / 2)
    bottom = y_center + int((y2 - y1) * crop_rate / 2)
    pick_coord = [left, top, right, bottom]

    mask_img = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.rectangle(mask_img, pick_coord[:2], pick_coord[2:], 255, -1)
    return mask_img