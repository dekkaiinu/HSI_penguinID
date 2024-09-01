import os
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra


NPYPATH = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/HS/npy/pix_vector/"
# W_NPYPATH = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/HS/npy/pix_vector/white/w"
# W_NPYPATH = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/RGB/npy/w"
# W_NPYPATH = '/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/HS/npy/flat/w'
# W_NPYPATH = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/HS/npy/smooth/ksize15/w"
W_NPYPATH = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/HS/npy/smooth/ksize9/w"

B_NPYPATH = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/HS/npy/pix_vector/black/b"

IMGJSON = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/images/images.json"
MASKJSON = "/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins/anotation/mask_images/mask_images.json"

CATEGORY_NUM = "/mnt/hdd1/youta/HSdataClassificationSystem/data/category_num.json"


# data ["bbox_id", "x", "class_id"]
def data_load(cfg_dataload):
    with open(CATEGORY_NUM, "r") as file:
        category_list = json.load(file)
    with open(MASKJSON, "r") as file:
        mask_list = json.load(file)
    dataset = []

    if cfg_dataload.pix == "white":
        pix_path = W_NPYPATH
    elif cfg_dataload.pix == "black":
        pix_path = B_NPYPATH
    else:
        pix_path = NPYPATH
    print("**************************************************")
    print("data loading")

    for i, category_info in tqdm(enumerate(category_list)):
        category = category_info["category_id"]
        if i >= cfg_dataload.class_num:
            break
        for mask_info in mask_list:
            mask_id = mask_info["mask_id"]
            label = mask_id.split("_")[1]
            if label == category:
                X = np.load(pix_path + mask_id + ".npy")
                y = i
                data = {}
                data["id"] = mask_id.split("_")[0]
                data["X"] = X
                data["y"] = y
                dataset.append(data)
    ids = []
    for data in dataset:
        ids.append(data["id"])
    
    unique_value = set(ids)
    unique_id_list = list(unique_value)
    id_dataset = []
    for unique_id in unique_id_list:
        Xandy_list = []
        for data in dataset:
            if data["id"] == unique_id:
                Xandy = {}
                Xandy["X"] = data["X"]
                Xandy["y"] = data["y"]
                Xandy_list.append(Xandy)
        id_data = {}
        id_data["id"] = unique_id
        id_data["data"] = Xandy_list
        id_dataset.append(id_data)
    
    id_dataset = sorted(id_dataset, key=lambda x: int(x['id']))
    return id_dataset

def data_load_idd(cfg_dataload):
    pkl_file_path = '/mnt/hdd1/youta/202402_IndividualDetection/dataset/ori0_dataset/ori0_data_imgid.pkl'
    with open(pkl_file_path, 'rb') as f:
        # pklデータを読み込む
        dataset_imgid = pickle.load(f)
    with open('/mnt/hdd1/youta/202402_IndividualDetection/hs_pixelwiseID/data/penguin_ids.json', "r") as file:
        category_list = json.load(file)
    category_ids = [item['category_id'] for item in category_list]
    

    if cfg_dataload.pix == "white":
        pix_path = W_NPYPATH
    elif cfg_dataload.pix == "black":
        pix_path = B_NPYPATH
    else:
        pix_path = NPYPATH
    
    id_dataset = []
    for img_id in tqdm(dataset_imgid):
        mask_img_paths = pix_path + img_id + '*'
        mask_img_files = glob.glob(mask_img_paths)

        Xandy_list = []
        for mask_img_file in mask_img_files:
            mask_img_id, _ = os.path.splitext(os.path.basename(mask_img_file))
            penguin_id = mask_img_id.split('_')[1]
            index_flag = False
            for category_id in category_ids:
                if category_id == penguin_id:
                    index_flag = True
            if index_flag == False:
                continue
            index = category_ids.index(penguin_id)

            Xandy = {}
            Xandy["X"] = np.load(pix_path + mask_img_id + ".npy")
            Xandy["y"] = index
            Xandy_list.append(Xandy)
        id_data = {}
        id_data['id'] = img_id
        id_data['data'] = Xandy_list
        id_dataset.append(id_data)
    
    return id_dataset
