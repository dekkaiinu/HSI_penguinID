import json
import csv
import re
import shutil
from yolo_format_data import yolo_format_data
from tqdm import tqdm


BBOXJSON = "//mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/anotation/bbox/bbox.json"
IMGJSON = "//mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/images.json"

IMGPATH = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/RGB/"

SAVEPATH = "/mnt/hdd1/youta/202402_IndividualDetection/dataset/ori0_dataset/YOLOdataset/"

TRAIN_CSV_PATH = '/mnt/hdd1/youta/202402_IndividualDetection/dataset/ori0_dataset/train.csv'
VAL_CSV_PATH = '/mnt/hdd1/youta/202402_IndividualDetection/dataset/ori0_dataset/val.csv'
TEST_CSV_PATH = '/mnt/hdd1/youta/202402_IndividualDetection/dataset/ori0_dataset/test.csv'


with open(IMGJSON, "r") as file:
    img_list = json.load(file)
with open(BBOXJSON, "r") as file:
    bbox_list = json.load(file)

with open(TRAIN_CSV_PATH, 'r') as file:
    reader = csv.reader(file)
    train_id = list(reader)[0]

with open(VAL_CSV_PATH, 'r') as file:
    reader = csv.reader(file)
    val_id = list(reader)[0]

with open(TEST_CSV_PATH, 'r') as file:
    reader = csv.reader(file)
    test_id = list(reader)[0]
# train_id = [re.search(r"\['(\d+)'\]", id_str).group(1) for id_str in train_id if re.search(r"\['(\d+)'\]", id_str)]
# print(train_id[0])
# exit()
with open('/mnt/hdd1/youta/202402_IndividualDetection/hs_pixelwiseID/data/penguin_ids.json', "r") as file:
    category_list = json.load(file)
category_ids = [item['category_id'] for item in category_list]


for i, img_info in tqdm(enumerate(img_list)):
    img_id = img_info["image_id"]

    train_flag = img_id in train_id
    val_flag = img_id in val_id
    test_flag = img_id in test_id

    if train_flag:
        save_label_path = SAVEPATH + "train/" + img_id + ".txt"
        save_img_path = SAVEPATH + "train/" + img_id + ".png"
    elif val_flag:
        save_label_path = SAVEPATH + "val/" + img_id + ".txt"
        save_img_path = SAVEPATH + "val/" + img_id + ".png"
    elif test_flag:
        save_label_path = SAVEPATH + "test/" + img_id + ".txt"
        save_img_path = SAVEPATH + "test/" + img_id + ".png"
    
    for bbox_info in bbox_list:
        img_id_bbox = bbox_info["image_id"]
        bbox = bbox_info["bbox"]
        penguin_id = bbox_info['category_id']

        if img_id == img_id_bbox:
            if penguin_id in category_ids:
                index = category_ids.index(penguin_id)
            else:
                index = 999
            text = yolo_format_data(bbox, cls_num=index)
            with open(save_label_path, "a") as file:
                file.write(text + "\n")
            shutil.copy(IMGPATH + img_id + ".png", save_img_path)

