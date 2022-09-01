import torch
import os
import pandas as pd
import shutil
import time
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torchcontrib.optim import SWA

from model import model as my_model
from data_loader import dataset as mydataset
from base.base_train import BaseTrain
from model.Metric import MeanRecall
from model.Metric import Acc
import model.confuse_matrix as cm_metrics

class MyTrainer(BaseTrain):
    def __init__(self, model, train_data, val_data, config, logger):
        super(MyTrainer, self).__init__(model, train_data, val_data, config, logger)

    def _train_epoch(self, cur_epoch):
        tic = time.time()

        self.model.train()
        train_loss = 0

        # Loop through each batch of training data
        correct = []
        predict = []

        pseudo_label =[]
        for _, (data, meta_data, target) in enumerate(self.train_data):
            meta_data = meta_data.float()
            data, meta_data, target = data.to(self.device), meta_data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data, meta_data)

            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            #pseudo_label = pseudo_label.extend(output.data.cpu().numpy().tolist())
            #temp = np.array(pseudo_label)
            #pseudo_label.extend()
            #print(temp.shape)
            _, pred = torch.max(output.data, 1)
            correct.extend(target.cpu().numpy())
            predict.extend(pred.cpu().numpy())
        #print(pseudo_label.shape)
#        self.optimizer.swap_swa_sgd()

        mean_acc = MeanRecall(predict, correct).get_mean_recall()
        acc = Acc(predict, correct).get_acc()



        # Update history and print metrics
        #self.train_history.update([1 - acc, 1 - val_acc])


        return train_loss, acc, mean_acc

    def _test_epoch(self, cur_epoch):
        self.model.eval()
        correct = []
        predict = []
        with torch.no_grad():
            for _, (data, meta_data, target) in enumerate(self.val_data):
                meta_data = meta_data.float()
                data, meta_data, target = data.to(self.device), meta_data.to(self.device), target.to(self.device)

                output = self.model(data, meta_data)
                _, pred = torch.max(output.data, 1)
                correct.extend(target.cpu().numpy())
                predict.extend(pred.cpu().numpy())
        # print(pseudo_label.shape)
        #        self.optimizer.swap_swa_sgd()
        cm_path = 'confuse_matrix_data/' + str(cur_epoch) + '_confusematrix'
        cm = cm_metrics.get_confuse_matrix(correct, predict)
        if not os.path.exists(cm_path):
            os.makedirs(cm_path)
        np.save(cm_path+'/cm_matrix.npy', cm)
        mean_recall = MeanRecall(predict, correct).get_mean_recall()
        acc = Acc(predict, correct).get_acc()
        return acc, mean_recall

    def train(self):

        # Nesterov accelerated gradient descent
        #optimizer = self.config.optimizer

        #Define our trainer for net
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        #self.optimizer = optim.Adam(trainable_params, lr=self.config.lr_rate)
        my_list = ['module.pretrained_net._fc.weight', 'module.pretrained_net._fc.bias',
                   'module.fc1.weight',
                   'module.fc1.bias',
                   'module.fc2.weight'
                   'module.fc2.bias',
                   'module.fc3.weight',
                   'module.fc3.bias'
                   ]
        params = []
        base_params = []
        for k, v in self.model.named_parameters():
            #print(k)
            if k in my_list:
                params.append(v)
            else:
                base_params.append(v)
        # #
        # params = list(filter(lambda k, v: k in my_list, self.model.named_parameters()))
        # base_params = list(filter(lambda kv: kv[1] not in my_list, self.model.named_parameters()))
        #print(params)
        base_opt = optim.SGD([
            {'params': base_params},
            {'params': params, 'lr': 10*self.config.lr_rate}
            ], lr=self.config.lr_rate, momentum=0.9, weight_decay=self.config.lr_decay, nesterov=True)
        self.optimizer = base_opt
        #self.optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
        #my_weight = torch.FloatTensor([2.8471, 1.0, 3.8745, 14.85, 4.9066, 53.8702, 50.8893, 20.5016]).cuda()
        class_weight = torch.Tensor( [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]).cuda()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        #self.loss_fn = nn.CrossEntropyLoss()
        epoList = [165, 220, 275]
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=epoList)
        for cur_epoch in range(1, self.config.epochs):

            train_loss, train_acc, train_mean_recall = self._train_epoch(cur_epoch)
            test_acc, test_mean_recall = self._test_epoch(cur_epoch)
            print('[Epoch %d] train_acc = %f  train_mean_recall = %f train_loss = %f test_acc = %f test_mean_recall = %f' %
                  (cur_epoch, train_acc, train_mean_recall, train_loss, test_acc, test_mean_recall))

            self.logger.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=cur_epoch)
            self.logger.add_scalar(tag='train_acc', scalar_value=train_acc, global_step=cur_epoch)
            self.logger.add_scalar(tag='train_mean_recall', scalar_value=train_mean_recall, global_step=cur_epoch)
            self.logger.add_scalar(tag='test_acc', scalar_value=test_acc, global_step=cur_epoch)
            self.logger.add_scalar(tag='test_mean_recall', scalar_value=test_mean_recall, global_step=cur_epoch)

            if not os.path.exists(self.config.model_path):
                os.makedirs(self.config.model_path)
            if cur_epoch % self.config.save_model_freq == 0:
                model_name = self.config.model_path + "/epoch%d_model.pt"%(cur_epoch)
                torch.save(self.model.state_dict(), model_name)

            scheduler.step(epoch=cur_epoch)
        self.logger.export_scalars_to_json("saved/logs/all_scalars.json")
        self.logger.close()


