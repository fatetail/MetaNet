import os
import pandas as pd
import shutil
import time
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from model import model as my_model
from data_loader import dataset as mydataset
from trainers import trainer as my_trainer
import configure.config as my_config

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

def main(config):
    tic = time.time()
    train_data_loader, val_data_loader = mydataset.get_dataloader(config)
    print('data_loader:time %f'%(time.time()-tic))
    tic = time.time()
    model = my_model.TwoD_Model(config.class_nums)
    print('model_loader:time %f'%(time.time()-tic))
    tic = time.time()
    logger = SummaryWriter(log_dir=config.log_path)
    print('log_loader:time %f'%(time.time()-tic))
    trainer = my_trainer.MyTrainer(model,
                      train_data =train_data_loader,
                      val_data =val_data_loader, config=config, logger=logger)

    trainer.train()


if __name__ == '__main__':
    args = my_config.parse_config()
    main(args)
