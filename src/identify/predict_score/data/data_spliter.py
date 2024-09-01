import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def data_spliter(cfg_spliter, dataset):
    if cfg_spliter.date_split:
        X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list, id = Xlsit_date_spliter(dataset)
        # X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list, id = Xlsit_time_spliter(dataset)

    else:
        X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list, id, train_dataset, val_dataset, test_dataset = Xlist_spliter(dataset, cfg_spliter)
    
    print("train_images = {}, val_images = {}, test_images = {}".format(len(X_train_list), len(X_val_list), len(X_test_list)))
    print("min image class number : train = {}, val = {}, test = {}".format(min_image_num(y_train_list), 
                                                                            min_image_num(y_val_list), 
                                                                            min_image_num(y_test_list)))
    

    return X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list, id, train_dataset, val_dataset, test_dataset

def min_image_num(y_list):
    unique_labels = set(y_list)
    image_num = []
    for label in unique_labels:
        label_indices = [i for i, x in enumerate(y_list) if x == label]
        image_num.append(len(label_indices))
    image_num = np.array(image_num)

    return np.min(image_num)

# def Xlist_spliter(dataset, cfg_spliter):
#     X = []
#     y = []
#     print("**************************************************")
#     print("train-test split procces")
#     for data in tqdm(dataset):
#         X.append([data["X"], data["id"]])
#         y.append(data["y"])
#     X_train, X_eva, y_train, y_eva = train_test_split(X, y, stratify=y, test_size=(1.0 - cfg_spliter.train_rate), random_state=1)
#     X_train, id = data_id_spliter(X_train)
#     X_eva, _ = data_id_spliter(X_eva)
#     X_val, X_test, y_val, y_test = train_test_split(X_eva, y_eva, stratify=y_eva, test_size=0.5)
#     return X_train, X_val, X_test, y_train, y_val, y_test, id


def Xlist_spliter(dataset, cfg_spliter):
    print("**************************************************")
    print("train-test split procces")
    
    random.seed(cfg_spliter.seed)
    random.shuffle(dataset)

    train_ratio = cfg_spliter.train_rate
    val_ratio = (1 - train_ratio) * 0.5

    total_samples = len(dataset)
    train_samples = int(train_ratio * total_samples)
    val_samples = int(val_ratio * total_samples)

    train_dataset = dataset[:train_samples]
    val_dataset = dataset[train_samples:train_samples+val_samples]
    test_dataset = dataset[train_samples+val_samples:]

    X_train, y_train, id_train = dataset_open(train_dataset)
    X_val, y_val, _ = dataset_open(val_dataset)
    X_test, y_test, _ = dataset_open(test_dataset)

    return X_train, X_val, X_test, y_train, y_val, y_test, id_train, train_dataset, val_dataset, test_dataset
        
def dataset_open(dataset):
    X = []
    y = []
    id = []
    for data in dataset:
        for Xandy in data["data"]:
            X.append(Xandy["X"])
            y.append(Xandy["y"])
            id.append(data["id"])
    return X, y, id

def data_id_spliter(X):
    splitedX = []
    id = []
    for x in X:
        splitedX.append(x[0])
        id.append(x[1])
    return splitedX, id

def Xlsit_date_spliter(dataset):
    X_train, X_eva = [], []
    y_train, y_eva = [], []
    ids = []
    print("**************************************************")
    print("train_test split process")
    for data in tqdm(dataset):
        id = data["id"]
        date = id[:8]
        for Xandy in data["data"]:
            X = Xandy["X"]
            y = Xandy["y"]
            if date == "20230623":
                X_train.append(X)
                y_train.append(y)
                ids.append(id)
            elif date == "20230627":
                X_eva.append(X)
                y_eva.append(y)
    X_val, X_test, y_val, y_test = train_test_split(X_eva, y_eva, stratify=y_eva, test_size=0.5, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test, ids

def Xlsit_time_spliter(dataset):
    X_train, X_eva = [], []
    y_train, y_eva = [], []
    ids = []
    print("**************************************************")
    print("train_test split process")
    for data in tqdm(dataset):
        id = data["id"]
        time = int(id[8:10])
        date = id[:8]
        for Xandy in data["data"]:
            X = Xandy["X"]
            y = Xandy["y"]
            # if (time > 12) & (date == '20230627'):
            #     X_train.append(X)
            #     y_train.append(y)
            #     ids.append(id)
            # elif (time <= 12) & (date == '20230627'):
            #     X_eva.append(X)
            #     y_eva.append(y)
            
            if (time <= 12) & (date == '20230627'):
                X_eva.append(X)
                y_eva.append(y)
            else:
                X_train.append(X)
                y_train.append(y)
                ids.append(id)
    X_val, X_test, y_val, y_test = train_test_split(X_eva, y_eva, stratify=y_eva, test_size=0.5, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test, ids