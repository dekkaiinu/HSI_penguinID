IMGPATH = "/mnt/hdd1/youta/202309_PenguinDetectWithYOLO/datasets_v5/penguin/val"

import cv2
import os

# 解像度を変更したいフォルダのパスを指定
folder_path = IMGPATH

# フォルダ内のファイルを走査
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        file_path = os.path.join(folder_path, filename)
        
        # 画像を開いて解像度を変更
        try:
            img = cv2.imread(file_path)

            # 元の画像のサイズを取得
            height, width, _ = img.shape

            # 新しい幅を640ピクセルに設定し、高さを比率を保ったままに計算
            new_width = 640
            new_height = int((new_width / width) * height)

            # 画像をリサイズ
            resized_img = cv2.resize(img, (new_width, new_height))
            cv2.imwrite(file_path, resized_img)  # 上書き保存
            print(f"変更完了: {filename}")
        except Exception as e:
            print(f"エラー: {filename} - {e}")