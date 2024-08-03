from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn as nn

from utils.create_runs_folder import *
from data.data_load import *
from data.data_spliter import data_spliter
from data.img_sampling import img_sampling
from data.pix_sampling import pix_sampling
from data.dataset_images2array import *
from data.data_prepro import data_prepro, zero_wave
from data.torch_data_loader import torch_data_loader
from data.noise_ref import noise_ref
from model_choice import *
from train import train
from test import test

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # runsに保存するためのpathとフォルダ作成
    # runs_path = create_runs_folder()

    # データの取り出しと，分割
    dataset = data_load_idd(cfg.dataload)
    X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list, id, train_dataset, val_dataset, test_dataset = data_spliter(cfg.split, dataset)

    save_dataset_path(train_dataset, 'train.csv')
    save_dataset_path(val_dataset, 'val.csv')
    save_dataset_path(test_dataset, 'test.csv')
    exit()

    if cfg.noise.flag:
        print("**************************************************")
        print("noise correction")
        print("pix_num=", cfg.noise.pix_num)
        X_train_list = noise_ref(X_train_list, cfg.noise.pix_num)
        X_val_list = noise_ref(X_val_list, cfg.noise.pix_num)
        X_test_list = noise_ref(X_test_list, cfg.noise.pix_num)


    # 取り出したデータのサンプリング
    if cfg.sampling.flag:
        if cfg.data_sampling.method == "img":
            X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list = img_sampling(X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list, cfg.data_sampling)
            X_train, X_val, X_test, y_train, y_val, y_test = dataset_images2array(X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list)
        elif cfg.data_sampling.method == "pix":
            X_train, X_val, X_test, y_train, y_val, y_test = dataset_images2array(X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list)
            X_train, X_val, X_test, y_train, y_val, y_test = pix_sampling(X_train, X_val, X_test, y_train, y_val, y_test, cfg.data_sampling)
    else: 
        X_train, X_val, X_test, y_train, y_val, y_test = dataset_images2array(X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list)

    # データの前処理
    X_train, X_val, X_test = data_prepro(X_train, X_val, X_test, cfg.prepro)


    # pytorch学習用のデータセット作成
    label_train_loader, label_val_loader, label_test_loader = torch_data_loader(X_train, X_val, X_test, y_train, y_val, y_test, cfg.train_parameter)

    # モデル，学習用のハイパラを定義
    input_dim = X_train.shape[1]
    output_dim = cfg.dataload.class_num
    model = model_choice(input_dim, output_dim, cfg.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # criterion
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_parameter.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=, gamma=)

    # 学習
    model = train(model, cfg.train_parameter.epochs, criterion, optimizer, label_train_loader, label_val_loader, runs_path)

    # 評価
    test(model, label_test_loader, criterion, optimizer, runs_path)

def save_dataset_path(dataset, save_file_name):
    save_path = '/mnt/hdd1/youta/202402_IndividualDetection/dataset/ori0_dataset/' + save_file_name

    img_id_list =[]
    for data in dataset:
        img_id = str(data['id'])
        img_id_list.append(img_id)
    import csv
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')  # delimiterを','に設定
        writer.writerow(img_id_list) 
        





if __name__ == "__main__":
    main()