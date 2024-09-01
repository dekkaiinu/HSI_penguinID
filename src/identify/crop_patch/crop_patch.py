import numpy as np
import cv2


def crop_patch(hsi: np.ndarray, bboxs: list, crop_size: int = 64) -> np.ndarray:
    """
    Crop a patch from a hyperspectral image based on bounding box coordinates.

    Args:
        hsi (np.ndarray): Hyperspectral image array with shape (height, width, channels).
        bboxs (list): Bounding box coordinates [x1, y1, x2, y2] normalized to [0, 1].
        crop_size (int, optional): Size of the square crop. Defaults to 64.

    Returns:
        np.ndarray: Cropped patch with shape (crop_size, crop_size, channels).
                    If the crop extends beyond the image boundaries, the result is padded with zeros.
    """
    center_x = int((bboxs[0] + bboxs[2]) / 2 * hsi.shape[1])
    center_y = int((bboxs[1] + bboxs[3]) / 2 * hsi.shape[0])

    start_x = max(0, center_x - crop_size // 2)
    end_x = min(hsi.shape[1], center_x + crop_size // 2)
    start_y = max(0, center_y - crop_size // 2)
    end_y = min(hsi.shape[0], center_y + crop_size // 2)
    
    cropped_hsi = hsi[start_y:end_y, start_x:end_x, :]
    
    if cropped_hsi.shape[0] < crop_size or cropped_hsi.shape[1] < crop_size:
        padded_hsi = np.zeros((crop_size, crop_size, hsi.shape[2]), dtype=hsi.dtype)
        padded_hsi[:cropped_hsi.shape[0], :cropped_hsi.shape[1], :] = cropped_hsi
        cropped_hsi = padded_hsi
    
    return cropped_hsi