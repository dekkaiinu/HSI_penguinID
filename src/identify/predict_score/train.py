from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from hspix_dataset import HsPixelDataset

from model_wrapper import ModelWrapper
from models.pix_wise_cnn import PointWiseCNN

from utils.create_runs_folder import *

@hydra.main(version_base=None, config_path='cfg', config_name='config')
def train(cfg: DictConfig):
    training_dataset = HsPixelDataset(feature_path=cfg.train.feature, target_path=cfg.train.target)
    training_dataset.balance_classes()

    # class_sample_count = np.array([len(np.where(training_dataset.target == t)[0]) for t in np.unique(training_dataset.target)])
    # weight = 1. / class_sample_count
    # samples_weight = np.array([weight[t] for t in training_dataset.target])
    # samples_weight = torch.from_numpy(samples_weight)
    # sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    training_dataset = DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)

    validation_dataset = HsPixelDataset(feature_path=cfg.validation.feature, target_path=cfg.validation.target)
    validation_dataset = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)


    # Init model
    model = PointWiseCNN(input_channels=cfg.input_dim, output_channels=cfg.output_dim, dropout_prob=cfg.dropout_prob)
    # Model to device
    model.to(cfg.device)
    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_parameter.lr)
    # Init learning rate schedule
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=cfg.train_parameter.epochs//10, 
                                                  gamma=cfg.train_parameter.scheduler_gamma)
    # Init loss function
    loss_function = nn.CrossEntropyLoss()
    # Init model wrapper
    model_wrapper = ModelWrapper(model=model,
                                 optimizer=optimizer,
                                 loss_function=loss_function,
                                 training_dataset=training_dataset,
                                 test_dataset=validation_dataset,
                                 lr_schedule=lr_schedule,
                                 device=cfg.device)
    # Perform training
    save_path = create_runs_folder()
    model_wrapper.training_process(epochs=cfg.train_parameter.epochs, save_path=save_path)

if __name__=='__main__':
    train()