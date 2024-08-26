import os
import numpy as np
import cv2

def hs2rgb(hsi: np.ndarray):
    hsi = hsi / 4095
    hsi = hsi.astype(np.float32)
    
    height, width = hsi.shape[0], hsi.shape[1]

    # hs2rgb.pyと同じディレクトリにあるcfm.csvを読み込む
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfm_path = os.path.join(script_dir, 'cfm.csv')
    color_matching_function = np.loadtxt(cfm_path, delimiter=',')
    color_matching_function = color_matching_function[::5]

    wave_length = np.arange(350, 1100 + 1, 5)

    index_low, index_hight = int(np.where(wave_length == color_matching_function[0, 0])[0]), int(np.where(wave_length == color_matching_function[-1, 0])[0]) + 1

    hsi_cie_range = hsi[:, :, index_low : index_hight]

    img_xyz = np.zeros((height, width, 3))
    img_rgb = np.zeros((height, width, 3))

    M = np.array([[0.41844, -0.15866, -0.08283],
                  [-0.09117, 0.25242, 0.01570],
                  [0.00092, -0.00255, 0.17858]])
    
    intensity = hsi_cie_range.reshape(-1, index_hight - index_low)
    
    xyz = np.dot(intensity, color_matching_function[:, 1:])

    img_xyz = xyz.reshape(height, width, 3)

    img_rgb = np.dot(img_xyz, M.T)

    img_rgb = (img_rgb - np.min(img_rgb)) / (np.max(img_rgb) - np.min(img_rgb)) * 255

    img_rgb = gamma(img_rgb)
    return img_rgb

def gamma(img_rgb):
    img_rgb = img_rgb.astype(np.float32) / np.max(img_rgb)
    img_rgb = img_rgb ** (1.0 / 2.2)
    img_rgb = img_rgb * 255
    img_rgb = img_rgb.astype(np.uint8)
    return img_rgb

def toneCurve1(frame, n = 1):
    look_up_table = np.zeros((256,1), dtype = 'uint8')
    for i in range(256):
        if i < 256 / n:
            look_up_table[i][0] = i * n
        else:
            look_up_table[i][0] = 255
    return cv2.LUT(frame, look_up_table)

def sToneCurve(frame):
    look_up_table = np.zeros((256,1), dtype = 'uint8')
    for i in range(256):
        look_up_table[i][0] = 255 * (np.sin(np.pi * (i/255 - 1/2)) + 1) / 2
    return cv2.LUT(frame, look_up_table)