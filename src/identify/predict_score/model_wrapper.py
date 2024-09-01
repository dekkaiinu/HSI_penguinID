from typing import Union, List, Any
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.output_metric import *
from utils.AvgrageMeter import *
from utils.accuracy import *
from utils.save_log import *
from utils.save_metrics import *
from utils.save_confusion_matrix import * 


class ModelWrapper(object):
    '''
    This class implements a model wrapper for training a classification model.
    '''
    def __init__(self,
                 model: Union[nn.Module, nn.DataParallel],
                 optimizer: torch.optim.Optimizer,
                 loss_function: nn.Module,
                 training_dataset: DataLoader,
                 test_dataset: DataLoader,
                 lr_schedule: Any,
                 device: str = 'cuda') -> None:
        '''
        Constructor method
        :param model: (Union[nn.Module, nn.DataParallel]) Model to be trained
        :param optimizer: (Optimizer) Optimizer module
        :param loss_function: (nn.Module) Loss function
        :param training_dataset: (DataLoader) Training dataset
        :param test_dataset: (DataLoader) Test dataset
        :param device: (str) Device to be utilized
        '''
        # Save parameters
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.lr_schedule = lr_schedule
        self.device = device

    def training_process(self, epochs: int, save_path):
        print('**************************************************')
        print('start training')
        tic = time.time()
        log_data = []

        for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
            self.optimizer.step()
            self.lr_schedule.step()
            # train
            self.model.train()
            train_acc, train_obj, tar_t, pre_t = self.train_epoch()
            OA1, AA_mean1, AA1 = output_metric(tar_t, pre_t)

            # val
            self.model.eval()
            val_acc, val_obj, tar_v, pre_v = self.valid_epoch(self.test_dataset)
            OA2, AA_mean2, AA2 = output_metric(tar_v, pre_v)

            tqdm.write('Epoch [{}/{}], train_loss: {:.4f} train_acc: {:.4f}, val_loss: {:.4f} val_acc: {:.4f}'
                            .format(epoch+1, epochs, train_obj, train_acc, val_obj, val_acc))

            log_entry = {
                'Epoch': epoch + 1,
                'Train Loss': np.round(train_obj.data.cpu().numpy(), 4),
                'Train Accuracy': np.round(train_acc.data.cpu().numpy(), 4),
                'Validation Loss': np.round(val_obj.data.cpu().numpy(), 4),
                'Validation Accuracy': np.round(val_acc.data.cpu().numpy(), 4)
            }
            log_data.append(log_entry)

            save_log(save_path + '/training_log.csv', log_data)
            torch.save(self.model.state_dict(), save_path + '/weight.pt')
        toc = time.time()

        cm_save(pre_t, tar_t, save_path + '/train/')
        save_metrics(OA1, AA_mean1, AA1, save_path + '/train/')
        cm_save(pre_v, tar_v, save_path + "/val/")
        save_metrics(OA2, AA_mean2, AA2, save_path + '/val/')

        print('Running Time: {:.2f}'.format(toc-tic))
        print('**************************************************')
        print('Final result:')
        print('OA: {:.4f} | AA: {:.4f}'.format(OA2, AA_mean2))
        print('**************************************************')

    def train_epoch(self):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        tar = np.array([])
        pre = np.array([])
        for batch_data, batch_target in tqdm(self.training_dataset, desc='Epoch', leave=False, total=len(self.training_dataset)):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()

            self.optimizer.zero_grad()
            batch_pred = self.model(batch_data)
            batch_pred = batch_pred.view(batch_pred.size(0), -1)
            
            loss = self.loss_function(batch_pred, batch_target)
            loss.backward()
            self.optimizer.step()

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())
        return top1.avg, objs.avg, tar, pre

    def valid_epoch(self, valid_dataset):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        tar = np.array([])
        pre = np.array([])
        for batch_data, batch_target in valid_dataset:
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()

            batch_pred = self.model(batch_data)
            batch_pred = batch_pred.view(batch_pred.size(0), -1)
            
            loss = self.loss_function(batch_pred, batch_target)

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())
        return top1.avg, objs.avg, tar, pre