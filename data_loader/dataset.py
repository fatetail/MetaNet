import torch
import numpy as np
import os
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import Augmentor
class MyDataset(Dataset):
    def __init__(self, csv_path, data_path, transform=None):
        self._csv_path = csv_path
        self._data_path = data_path
        self._transform = transform
        self._path_list, self._exter_data, self._label = self._list_data()

    # 保存video的名字列表，video的label
    def _list_data(self):
        df = pd.read_csv(self._csv_path)
        np_data = np.array(df['image_id'].values)
        for item_index in range(len(np_data)):
            np_data[item_index] = self._data_path + np_data[item_index] + '.jpg'
        label = df['label'].values
        np_external_data = np.array(df.values[:, 3:], dtype='float')
        print(label.shape)
        return np_data, np_external_data, label

    def __getitem__(self, idx):
        img_path = self._path_list[idx]
        #print(img_path)
        img = Image.open(img_path)
        if self._transform is not None:
            img = self._transform(img)
        return (img, self._exter_data[idx], self._label[idx])

    def __len__(self):
        return len(self._label)

def get_dataloader(config):
    p = Augmentor.Pipeline()
    p.torch_transform()
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.skew(probability=0.5)
    p.random_brightness(probability=0.5, min_factor=0.8, max_factor=0.9)
    p.random_color(probability=0.5, min_factor=0.8, max_factor=0.9)
    p.random_contrast(probability=0.5, min_factor=0.8, max_factor=0.9)
    p.rotate_random_90(probability=0.5)
    #p.shear(0.5, 5, 5)

    normMean = (0.6680, 0.5297, 0.5247)
    normStd = (0.1894, 0.1823, 0.1971)

    transform_train = transforms.Compose([
        # 将图像放大成高和宽各为40像素的正方形
        # gdata.vision.transforms.Resize((2)),
        # 随机对高和宽各为40像素的正方形图像裁剪出面积为原图像面积0.64~1倍的小正方形，再放缩为
        # 高和宽各为32像素的正方形
        #transforms.Resize((300, 300)),
        transforms.RandomCrop((224, 224)),
        p.torch_transform(),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),

        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),

        transforms.ToTensor(),
        # 对图像的每个通道做标准化
        # transforms.Normalize(mean=[0.6238, 0.5201, 0.5039],
        #                      std=[0.2420, 0.2234, 0.2312])
        transforms.Normalize(mean=normMean,
                             std=normStd)
                             ])

    transform_val = transforms.Compose([
        #transforms.Resize(size=(300, 300)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.6238, 0.5201, 0.5039],
        #                      std=[0.2420, 0.2234, 0.2312])
        #                     ]
        transforms.Normalize(mean=normMean,
                             std=normStd)
    ])


    train_csv_path = 'csv_file/train_fold_1.csv'
    val_csv_path = 'csv_file/val_fold_1.csv'

    train_data_path = '/data0/weipeng/ISIC2018_Data/ISIC_Data/train_val_300x300/'
    test_data_path = '/data0/weipeng/ISIC2018_Data/ISIC_Data/train_val_300x300/'


    train_data = DataLoader(MyDataset(train_csv_path, train_data_path, transform_train),
                            batch_size=config.batch_size, shuffle=True, num_workers=2)

    val_data = DataLoader(MyDataset(val_csv_path, test_data_path, transform_val),
                            batch_size=4*config.batch_size, shuffle=False, num_workers=2)
    return train_data, val_data
