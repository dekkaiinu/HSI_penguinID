import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List


def plot_rectangle(image: np.ndarray,
                        bboxes_list: List[np.ndarray],
                        labels: List[int] = None,
                        label_size: int = 7,
                        line_width: int = 1,
                        border_color=(0, 1, 0, 1)) -> None:
    """
    画像にnumpy配列リストから矩形を描画する関数

    Args:
        image: 画像データ（NumPy配列）
        bboxes_list: numpy配列リスト（各要素は矩形情報）
        labels: 矩形ラベルリスト（オプション）
        label_size: ラベル文字サイズ（デフォルト:7）
        line_width: 矩形線の太さ（デフォルト:1）
        border_color: 矩形線のカラー（デフォルト:緑）
    """
    
    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(image)

    if labels is None:
        labels = [None] * len(bboxes_list)

    for bbox, label in zip(bboxes_list, labels):
        # Add bounding box
        label = str(label)
        x1 = int(bbox[0] * image.shape[1])
        x2 = int(bbox[2] * image.shape[1])
        y1 = int(bbox[1] * image.shape[0])
        y2 = int(bbox[3] * image.shape[0])

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=line_width,
                                 edgecolor=border_color,
                                 facecolor='none')
        ax.add_patch(rect)

        # label
        if label:
            bbox_props = dict(boxstyle="square,pad=0",
                              linewidth=line_width, facecolor=border_color,
                              edgecolor=border_color)
            ax.text(x1, y1, label,
                    ha="left", va="bottom", rotation=0,
                    size=label_size, bbox=bbox_props)
    ax.axis('off')

    buf = io.BytesIO() # bufferを用意
    plt.savefig(buf, dpi=200, bbox_inches='tight', pad_inches=0, transparent=True)
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
    ploted_img = cv2.imdecode(enc, 1) # デコード
    plt.clf()
    return ploted_img