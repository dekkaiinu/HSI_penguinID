import numpy as np

def gamma(img_rgb):
    img_rgb = img_rgb.astype(np.float32) / np.max(img_rgb)
    img_rgb = img_rgb ** (1.0 / 2.2)
    img_rgb = img_rgb * 255
    img_rgb = img_rgb.astype(np.uint8)
    return img_rgb