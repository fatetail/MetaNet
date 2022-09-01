import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd, gluon, init
from mxnet.gluon import data as gdata, loss as gloss, nn
import os
import pandas as pd
import shutil
import time
import numpy as np
from tqdm import tqdm
from model import model as my_model
from gluoncv.utils import makedirs, TrainingHistory
from data_loader import dataset as mydataset
from mxnet import autograd as ag
from base.base_test import BaseTest


class MyTester(BaseTest):
    def __init__(self, model, test_data, config, logger):
        super(MyTester, self).__init__(model, test_data, config, logger)
        if self.config.is_gpu:
            self.ctx = [mx.gpu(i) for i in range(self.config.start_gpu_idx, self.config.end_gpu_idx + 1)]
        else:
            self.ctx = [mx.cpu(0)]

        self.model.collect_params().reset_ctx(self.ctx)

        for dirpath, dirnames, files in os.walk(self.config.model_path):
            if files:
                model_path = os.path.join(dirpath, 'epoch30_model.params')
                print(model_path)
                self.model.load_parameters(model_path, self.ctx)

    def _test_epoch(self):
        metric = mx.metric.Accuracy()
        for batch in tqdm(self.test_data):
            data = batch[0].as_in_context(self.ctx[0])
            label = batch[1].as_in_context(self.ctx[0])
            outputs = self.model(data)
            item = nd.argmax(outputs, axis=1)
            print(item)
            metric.update(label, outputs)

        return metric.get()

    def test(self):

        self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        self.train_metric = mx.metric.Accuracy()

        tic = time.time()
        name, test_acc = self._test_epoch()
        print('test_acc=%f time: %f' %
              (test_acc, time.time() - tic))
        self.logger.add_scalar(tag='test_acc', value=test_acc)
        self.logger.close()


