import numpy as np
import cv2

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